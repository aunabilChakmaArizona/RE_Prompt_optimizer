#!/usr/bin/env python3
# etgpo_full.py

"""
Unified Prompt Optimization Comparison using DSPy

Compares three prompt optimization methods on equal footing:
1. Taxonomy-guided prompt optimization (our method)
2. GEPA (Reflective Prompt Evolution)
3. MIPROv2 (Multi-prompt Instruction Proposal Optimizer)

All methods use DSPy's evaluation infrastructure for fair comparison.

V2 Changes:
- Factored execution: Can load a saved taxonomy and only re-run guidance generation + evaluation
- Category-level tracking: Track per-problem, per-category accuracy before/after optimization
- Ablation mode: Streamlined execution for hyperparameter ablation studies

Usage:
    # Run all methods
    python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy,gepa,mipro

    # Run only taxonomy vs baseline
    python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy

    # Factored execution: Load existing taxonomy and vary T/G
    python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy \
        --load_taxonomy path/to/taxonomy.json --coverage_threshold 0.5 --max_guidances 5

    # Ablation mode: Skip test eval, focus on validation
    python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy \
        --ablation_mode --taxonomy_runs 5 --coverage_threshold 1.0 --max_guidances 10
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from copy import deepcopy

CURRENT_DIR = Path(__file__).resolve().parent
CODES_DIR = CURRENT_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.append(str(CODES_DIR))

import dspy
from dspy import GEPA, MIPROv2
from scipy.stats import binom

from common.data_utils import get_validation_examples, get_test_examples
from common.answer_equivalence import judge_answer
from common.llm_interface import get_json_response_from_gpt
from common.config import AGENT_MODEL
from common.token_tracker import token_tracker, format_cost
from agents.agent_data_loader import load_train_samples
from agents.agent_data_loader import load_split_episodes
from agents.agent_binary_inference import run_binary_inference, _resolve_binary_token_id
from agents.agent_evaluate import evaluate_fn as evaluate_re_episode_fn
from agents.agent_feedback_samples import FeedbackSample, FeedbackSamples
from agents.agent_generate_feedback import generate_feedback_fn as generate_re_feedback_fn
from agents.agent_graph_node import GraphNode
from agents.agent_llm_prompting import run_prompt
from agents.agent_metrics import compute_prf_stats
from agents.agent_models import load_model_and_tokenizer
from agents.agent_prompts import (
    FEEDBACK_PROMPT_MAP,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
)
from agents.agent_relation_utils import get_relation_description
from agents.agent_run_inference import _build_inference_prompt as build_re_inference_prompt
from agents.agent_scorer import NO_RELATION
from agents.agent_sample_feedback import sample_feedback_fn
from agents.agent_train_pipeline import build_root_node, enrich_feedback_samples


# =============================================================================
# Constants
# =============================================================================

# API key file paths
MAIN_API_KEY_FILE = "/workspace/openai"
# MAIN_API_KEY_FILE = "/workspace/together"
REFLECTION_API_KEY_FILE = "/workspace/openai"

# Default models - can be overridden via CLI
DEFAULT_MAIN_MODEL = "openai/gpt-4.1-mini"
# DEFAULT_MAIN_MODEL = "together_ai/deepseek-ai/DeepSeek-V3.1"
# DEFAULT_MAIN_MODEL = "together_ai/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MAIN_TEMPERATURE = 1.0
DEFAULT_MAIN_MAX_TOKENS = 32768

DEFAULT_REFLECTION_MODEL = "openai/gpt-4.1"
DEFAULT_REFLECTION_TEMPERATURE = 0.1
DEFAULT_REFLECTION_MAX_TOKENS = 32768

DEFAULT_TAXONOMY_MODEL = AGENT_MODEL  # For taxonomy analysis and guidance generation

# Base instruction for CoT
BASE_COT_INSTRUCTION = "Please think step by step and then solve the task."
RE_TASK_MODE = "relation_extraction_non_reasoning"

# Generic error types
GENERIC_ERROR_TYPES = "Calculation Error, Wrong Approach, Conceptual Misunderstanding, Missing Step, Logical Fallacy, Factual Error, Incomplete Reasoning, Misreading the Problem"


# =============================================================================
# Domain Configuration
# =============================================================================

DOMAIN_CONFIGS = {
    "math": {"domain_description": "math problems"},
    "science_mcq": {"domain_description": "graduate-level science questions"},
    "mcq": {"domain_description": "multiple choice reasoning questions"},
    "multi_hop_qa": {"domain_description": "multi-hop reasoning questions"},
    "logical_reasoning": {"domain_description": "logical reasoning questions"},
}

DATASET_TO_DOMAIN = {
    "math": "math",
    "math_sol": "math",
    "aime": "math",
    "aimo": "math",
    "aimo_val_split": "math",
    "hmmt_nov_2025": "math",
    "hmmt_feb_2025": "math",
    "gpqa": "science_mcq",
    "mmlupro": "mcq",
    "musique": "multi_hop_qa",
    "hotpotqa": "multi_hop_qa",
    "ar_lsat": "logical_reasoning",
    "folio": "logical_reasoning",
}


def get_domain_description(dataset: str) -> str:
    """Get the domain description for a dataset."""
    domain = DATASET_TO_DOMAIN.get(dataset, "math")
    config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["math"])
    return config["domain_description"]


# =============================================================================
# JSON Encoder for NumPy Types
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Token Tracking Utilities - DSPy LM History
# =============================================================================

def get_token_counts_from_history(lm) -> Dict[str, int]:
    """Extract total input/output tokens from DSPy LM history."""
    input_tokens = 0
    output_tokens = 0
    
    if not hasattr(lm, 'history') or lm.history is None:
        return {"input_tokens": 0, "output_tokens": 0}
    
    for entry in lm.history:
        if isinstance(entry, dict):
            usage = entry.get('usage', {})
            if usage:
                input_tokens += usage.get('prompt_tokens', 0) or 0
                output_tokens += usage.get('completion_tokens', 0) or 0
    
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def clear_lm_history(lm, label: str = ""):
    """Clear the DSPy LM history to start fresh tracking."""
    if hasattr(lm, 'history'):
        lm.history = []
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Cleared LM history.")


# =============================================================================
# Token Tracking Utilities - token_tracker (for taxonomy/guidance phases)
# =============================================================================

def capture_token_snapshot() -> Dict[str, Dict]:
    """Capture current state from token_tracker."""
    _, breakdown = token_tracker.get_aggregated_costs()
    return deepcopy(breakdown)


def compute_phase_delta(before: Dict[str, Dict], after: Dict[str, Dict]) -> Dict[str, Dict]:
    """Compute the difference between two token snapshots."""
    delta = {}
    all_models = set(before.keys()) | set(after.keys())
    
    for model in all_models:
        before_stats = before.get(model, {
            "input_tokens": 0, "output_tokens": 0, "input_cost": 0.0,
            "output_cost": 0.0, "total_cost": 0.0, "calls": 0
        })
        after_stats = after.get(model, {
            "input_tokens": 0, "output_tokens": 0, "input_cost": 0.0,
            "output_cost": 0.0, "total_cost": 0.0, "calls": 0
        })
        
        delta[model] = {
            "input_tokens": after_stats.get("input_tokens", 0) - before_stats.get("input_tokens", 0),
            "output_tokens": after_stats.get("output_tokens", 0) - before_stats.get("output_tokens", 0),
            "input_cost": after_stats.get("input_cost", 0.0) - before_stats.get("input_cost", 0.0),
            "output_cost": after_stats.get("output_cost", 0.0) - before_stats.get("output_cost", 0.0),
            "total_cost": after_stats.get("total_cost", 0.0) - before_stats.get("total_cost", 0.0),
            "calls": after_stats.get("calls", 0) - before_stats.get("calls", 0)
        }
    
    delta = {model: stats for model, stats in delta.items() if stats["calls"] > 0}
    return delta


def convert_dspy_history_to_phase_stats(lm, model_name: str) -> Dict[str, Dict]:
    """Convert DSPy LM history to phase stats format."""
    token_counts = get_token_counts_from_history(lm)
    
    if token_counts["input_tokens"] == 0 and token_counts["output_tokens"] == 0:
        return {}
    
    call_count = len(lm.history) if hasattr(lm, 'history') and lm.history else 0
    
    return {
        model_name: {
            "input_tokens": token_counts["input_tokens"],
            "output_tokens": token_counts["output_tokens"],
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": 0.0,
            "calls": call_count
        }
    }


# =============================================================================
# Token Tracking Utilities - Printing Functions
# =============================================================================

def print_phase_token_stats(phase_name: str, stats: Dict[str, Dict]) -> None:
    """Print formatted token statistics for a single phase."""
    if not stats:
        print(f"\n📊 PHASE: {phase_name.upper()}")
        print("──────────────────────────────────────")
        print("  (no LLM calls in this phase)")
        return
    
    print(f"\n📊 PHASE: {phase_name.upper()}")
    print("──────────────────────────────────────")
    
    phase_total_input = 0
    phase_total_output = 0
    phase_total_cost = 0.0
    phase_total_calls = 0
    
    for model, model_stats in sorted(stats.items()):
        input_tokens = model_stats.get("input_tokens", 0)
        output_tokens = model_stats.get("output_tokens", 0)
        input_cost = model_stats.get("input_cost", 0.0)
        output_cost = model_stats.get("output_cost", 0.0)
        total_cost = model_stats.get("total_cost", 0.0)
        calls = model_stats.get("calls", 0)
        
        phase_total_input += input_tokens
        phase_total_output += output_tokens
        phase_total_cost += total_cost
        phase_total_calls += calls
        
        print(f"  {model}:")
        print(f"    Input:    {input_tokens:>12,} tokens ({format_cost(input_cost)})")
        print(f"    Output:   {output_tokens:>12,} tokens ({format_cost(output_cost)})")
        print(f"    Calls:    {calls:>12,}")
    
    print("──────────────────────────────────────")
    print(f"  Phase Total:")
    print(f"    Input:    {phase_total_input:>12,} tokens")
    print(f"    Output:   {phase_total_output:>12,} tokens")
    print(f"    Cost:     {format_cost(phase_total_cost)}")
    print(f"    Calls:    {phase_total_calls:>12,}")


def print_final_token_summary(all_phase_stats: Dict[str, Dict[str, Dict]]) -> None:
    """Print grand total token summary."""
    print("\n" + "=" * 60)
    print("📊 FINAL TOKEN USAGE SUMMARY")
    print("=" * 60)
    
    model_totals: Dict[str, Dict] = defaultdict(lambda: {
        "input_tokens": 0, "output_tokens": 0, "input_cost": 0.0,
        "output_cost": 0.0, "total_cost": 0.0, "calls": 0
    })
    
    phase_totals: Dict[str, Dict] = {}
    
    for phase_name, phase_stats in all_phase_stats.items():
        phase_input = 0
        phase_output = 0
        phase_cost = 0.0
        phase_calls = 0
        
        for model, model_stats in phase_stats.items():
            model_totals[model]["input_tokens"] += model_stats.get("input_tokens", 0)
            model_totals[model]["output_tokens"] += model_stats.get("output_tokens", 0)
            model_totals[model]["input_cost"] += model_stats.get("input_cost", 0.0)
            model_totals[model]["output_cost"] += model_stats.get("output_cost", 0.0)
            model_totals[model]["total_cost"] += model_stats.get("total_cost", 0.0)
            model_totals[model]["calls"] += model_stats.get("calls", 0)
            
            phase_input += model_stats.get("input_tokens", 0)
            phase_output += model_stats.get("output_tokens", 0)
            phase_cost += model_stats.get("total_cost", 0.0)
            phase_calls += model_stats.get("calls", 0)
        
        phase_totals[phase_name] = {
            "input_tokens": phase_input, "output_tokens": phase_output,
            "total_cost": phase_cost, "calls": phase_calls
        }
    
    print("\n📈 BY PHASE:")
    print("──────────────────────────────────────")
    
    grand_input = 0
    grand_output = 0
    grand_cost = 0.0
    grand_calls = 0
    
    for phase_name, totals in all_phase_stats.items():
        phase_total = phase_totals.get(phase_name, {})
        input_tok = phase_total.get("input_tokens", 0)
        output_tok = phase_total.get("output_tokens", 0)
        cost = phase_total.get("total_cost", 0.0)
        calls = phase_total.get("calls", 0)
        
        grand_input += input_tok
        grand_output += output_tok
        grand_cost += cost
        grand_calls += calls
        
        print(f"\n  {phase_name}:")
        print(f"    Total Tokens: {input_tok + output_tok:>12,} ({input_tok:,} in / {output_tok:,} out)")
        print(f"    Total Cost:   {format_cost(cost)}")
        print(f"    Total Calls:  {calls:>12,}")
    
    print("\n" + "=" * 60)
    print("📊 GRAND TOTAL:")
    print(f"    Input:    {grand_input:>12,} tokens")
    print(f"    Output:   {grand_output:>12,} tokens")
    print(f"    Total:    {grand_input + grand_output:>12,} tokens")
    print(f"    Cost:     {format_cost(grand_cost)}")
    print(f"    Calls:    {grand_calls:>12,}")
    print("=" * 60)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FailureRecord:
    """A single failure trace with metadata."""
    problem_idx: int
    run_id: int
    problem: str
    correct_answer: str
    predicted_answer: str
    reasoning: str
    solution: Optional[str] = None


@dataclass
class REPairExample:
    """A TACRED-style support/query pair for binary relation inference."""
    problem_idx: int
    relation: str
    support_sentence: str
    query_sentence: str
    answer: str
    relation_description: str
    id_1shot: str = ""
    id_query: str = ""


@dataclass
class IssueCategory:
    """A category of issue found in failure traces."""
    category_name: str
    summary: str
    description: str
    example: str
    error_type: str
    why_leads_to_wrong_answer: str
    trace_count: int = 0


@dataclass
class TraceAssignment:
    """Maps a specific trace to an issue category."""
    problem_idx: int
    run_id: int
    category_name: str
    trace_specific_location: str
    trace_specific_details: str


@dataclass
class CategoryStats:
    """Statistics for a single category."""
    category_name: str
    failure_count: int
    problem_count: int
    problem_list: List[int]
    coverage_fraction: float


@dataclass
class MethodResult:
    """Results for a single method evaluation."""
    method_name: str
    per_run_accuracies: List[float]
    mean_accuracy: float
    ci_lower: float
    ci_upper: float
    std: float
    min_acc: float
    max_acc: float
    num_runs: int
    detailed_results: List[Tuple[int, int, bool]]


@dataclass
class CategoryLevelResult:
    """Per-category accuracy results for category-level analysis."""
    category_name: str
    problem_indices: List[int]
    num_problems: int
    baseline_failure_rate: float
    taxonomy_failure_rate: float
    delta_failure_rate: float
    was_selected: bool
    baseline_failures: int
    taxonomy_failures: int
    total_evals: int


# =============================================================================
# DSPy Signature Factory
# =============================================================================

def create_signature_class(instruction: str, name: str = "ReasoningTask"):
    """Dynamically create a DSPy Signature class with custom instruction."""
    class_dict = {
        "__doc__": instruction,
        "__module__": __name__,
        "problem": dspy.InputField(),
        "reasoning": dspy.OutputField(desc="Step-by-step reasoning"),
        "answer": dspy.OutputField(desc="Final answer"),
    }
    return type(name, (dspy.Signature,), class_dict)


# =============================================================================
# Metric Functions
# =============================================================================

def create_metric(dataset_type: str):
    """Create an evaluation metric function for the given dataset type."""
    def metric(example, prediction, trace=None):
        ground_truth = str(example.answer)
        predicted_answer = str(prediction.answer) if hasattr(prediction, 'answer') else str(prediction)
        question = str(example.problem)
        
        try:
            is_correct = judge_answer(
                ground_truth=ground_truth,
                final_answer=predicted_answer,
                question=question,
                dataset_type=dataset_type
            )
            return int(is_correct)
        except Exception as e:
            print(f"Warning: judge_answer failed: {e}")
            return 0
    
    return metric


def create_metric_with_feedback(dataset_type: str):
    """Create a metric with feedback function for GEPA optimization."""
    def metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
        ground_truth = str(example.answer)
        predicted_answer = str(prediction.answer) if hasattr(prediction, 'answer') else str(prediction)
        question = str(example.problem)
        written_solution = getattr(example, 'solution', '') or ''
        
        try:
            is_correct = judge_answer(
                ground_truth=ground_truth,
                final_answer=predicted_answer,
                question=question,
                dataset_type=dataset_type
            )
            score = int(is_correct)
        except Exception as e:
            print(f"Warning: judge_answer failed: {e}")
            score = 0

        feedback_text = ""
        if score == 1:
            feedback_text = f"Your answer is correct. The correct answer is '{ground_truth}'."
        else:
            feedback_text = f"Your answer is incorrect. You answered '{predicted_answer}', but the correct answer is '{ground_truth}'."
        
        if written_solution:
            feedback_text += f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution."

        return dspy.Prediction(score=score, feedback=feedback_text)
    
    return metric_with_feedback


# =============================================================================
# Multi-Run Evaluation
# =============================================================================

def run_multi_run_evaluation(
    program: dspy.Module,
    dataset: List[dspy.Example],
    metric,
    num_runs: int,
    num_threads: int = 64,
    display_progress: bool = True
) -> Tuple[List[float], List[Tuple[int, int, bool]]]:
    """Run evaluation multiple times by expanding dataset."""
    if num_runs <= 0:
        print(f"\n  Skipping evaluation: num_runs = {num_runs}")
        return [], []
    
    num_problems = len(dataset)
    
    if num_problems == 0:
        print(f"\n  Skipping evaluation: empty dataset")
        return [], []
    
    total_examples = num_problems * num_runs
    
    print(f"\n  Expanding {num_problems} problems × {num_runs} runs = {total_examples} evaluations...")
    
    expanded_dataset = [ex for ex in dataset for _ in range(num_runs)]
    
    evaluator = dspy.Evaluate(
        devset=expanded_dataset,
        metric=metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=True,
        provide_traceback=True,
    )
    
    eval_result = evaluator(program)
    
    detailed_results = []
    per_run_correct = [0] * num_runs
    
    for idx, (example, prediction, score) in enumerate(eval_result.results):
        prob_idx = idx // num_runs
        run_id = idx % num_runs
        is_correct = bool(score > 0)
        detailed_results.append((prob_idx, run_id, is_correct))
        if is_correct:
            per_run_correct[run_id] += 1
    
    per_run_accuracies = [correct / num_problems if num_problems > 0 else 0.0 
                          for correct in per_run_correct]
    
    mean_acc = sum(per_run_accuracies) / len(per_run_accuracies) if per_run_accuracies else 0.0
    min_acc = min(per_run_accuracies) if per_run_accuracies else 0.0
    max_acc = max(per_run_accuracies) if per_run_accuracies else 0.0
    print(f"  Mean accuracy: {mean_acc * 100:.2f}% (min: {min_acc * 100:.2f}%, max: {max_acc * 100:.2f}%)")
    
    return per_run_accuracies, detailed_results


# =============================================================================
# Statistical Functions
# =============================================================================

def compute_bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for mean."""
    if not values:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    mean = np.mean(values)
    
    if len(values) < 2:
        return float(mean), float(mean), float(mean)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(mean), float(lower), float(upper)


def compute_per_run_stats(per_run_accuracies: List[float]) -> Dict[str, Any]:
    """Compute comprehensive per-run statistics with bootstrap CI."""
    if not per_run_accuracies:
        return {
            "mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "std": 0.0, "min": 0.0, "max": 0.0, "num_runs": 0
        }
    
    accuracies = np.array(per_run_accuracies)
    mean, ci_lower, ci_upper = compute_bootstrap_ci(per_run_accuracies)
    
    return {
        "mean": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0,
        "min": float(np.min(accuracies)),
        "max": float(np.max(accuracies)),
        "num_runs": len(accuracies)
    }


def compute_paired_difference_ci(
    baseline_per_run: List[float],
    method_per_run: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """Compute bootstrap CI for paired difference (method - baseline)."""
    if len(baseline_per_run) != len(method_per_run):
        raise ValueError("Baseline and method must have same number of runs")
    
    if len(baseline_per_run) == 0:
        return {
            "mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "std_diff": 0.0, "num_runs": 0
        }
    
    diffs = np.array([m - b for b, m in zip(baseline_per_run, method_per_run)])
    mean_diff, ci_lower, ci_upper = compute_bootstrap_ci(list(diffs), confidence, n_bootstrap)
    
    return {
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "num_runs": len(diffs)
    }


def mcnemar_test(
    baseline_results: List[Tuple[int, int, bool]],
    method_results: List[Tuple[int, int, bool]],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """McNemar's test for paired binary outcomes."""
    baseline_dict = {(p, r): c for p, r, c in baseline_results}
    method_dict = {(p, r): c for p, r, c in method_results}
    
    b = 0  # baseline correct, method wrong
    c = 0  # baseline wrong, method correct
    concordant_both_right = 0
    concordant_both_wrong = 0
    
    all_keys = set(baseline_dict.keys()) & set(method_dict.keys())
    
    for key in all_keys:
        base_correct = baseline_dict[key]
        method_correct = method_dict[key]
        
        if base_correct and not method_correct:
            b += 1
        elif not base_correct and method_correct:
            c += 1
        elif base_correct and method_correct:
            concordant_both_right += 1
        else:
            concordant_both_wrong += 1
    
    total_discordant = b + c
    total_cells = b + c + concordant_both_right + concordant_both_wrong
    
    if total_discordant == 0:
        return {
            "is_significant": False, "p_value": 1.0, "b": b, "c": c,
            "total_discordant": total_discordant, "total_cells": total_cells,
            "improvement_rate": 0.0
        }
    
    p_value = binom.sf(c - 1, total_discordant, 0.5)
    improvement_rate = (c - b) / total_cells if total_cells > 0 else 0.0
    
    return {
        "is_significant": (p_value < alpha) and (c > b),
        "p_value": float(p_value),
        "b": b, "c": c,
        "total_discordant": total_discordant,
        "total_cells": total_cells,
        "improvement_rate": improvement_rate
    }


def format_accuracy_with_ci(stats: Dict[str, Any]) -> str:
    """Format accuracy with CI for display."""
    mean = stats["mean"] * 100
    ci_lower = stats["ci_lower"] * 100
    ci_upper = stats["ci_upper"] * 100
    std = stats["std"] * 100
    return f"{mean:.2f}% [{ci_lower:.2f}%, {ci_upper:.2f}%] (std={std:.2f}%)"


def format_diff_with_ci(diff_stats: Dict[str, Any]) -> str:
    """Format difference with CI for display."""
    mean = diff_stats["mean_diff"] * 100
    ci_lower = diff_stats["ci_lower"] * 100
    ci_upper = diff_stats["ci_upper"] * 100
    return f"{mean:+.2f}% [{ci_lower:+.2f}%, {ci_upper:+.2f}%]"


# =============================================================================
# Taxonomy Trace Formatting
# =============================================================================

def format_failure_for_analysis(
    failure: FailureRecord,
    reasoning_label: str = "Model's Reasoning"
) -> str:
    """Format a failure record for taxonomy LLM analysis."""
    solution_section = ""
    if failure.solution:
        solution_section = f"""
### Correct Solution
{failure.solution}
"""
    
    return f"""
## Failure (Problem {failure.problem_idx}, Run {failure.run_id})

### Problem
{failure.problem}

### Correct Answer
{failure.correct_answer}
{solution_section}

### {reasoning_label}
{failure.reasoning}

### Model's Answer
{failure.predicted_answer}

---
"""


# =============================================================================
# Taxonomy Building Prompts
# =============================================================================

def generate_first_batch_taxonomy_prompt(
    failures: List[FailureRecord],
    include_solution: bool = False,
    domain_description: str = "math problems",
    reasoning_label: str = "Model's Reasoning",
    reasoning_source_note: str = ""
) -> str:
    """Generate prompt for analyzing the first batch of failures."""
    
    failures_text = ""
    for i, failure in enumerate(failures, 1):
        failures_text += f"\n## Failure {i}"
        failures_text += format_failure_for_analysis(failure, reasoning_label=reasoning_label)
    
    solution_instruction = ""
    if include_solution:
        solution_instruction = "\nCompare the model's reasoning to the correct solution to identify where it diverged."
    
    return f"""You are an expert at analyzing why language models fail on {domain_description}.

{failures_text}

## Your Task

Analyze each failure and identify the root cause of each error. Be as descriptive as possible.{solution_instruction}
{reasoning_source_note}

For each failure, find:
1. The EARLIEST point in the reasoning where something went wrong
2. What specifically went wrong (calculation error, wrong approach, misunderstanding, etc.)
3. Why this error led to the wrong final answer

Create issue categories that capture each type of error. Categories should be general enough to potentially apply to other traces, but specific enough to be meaningful.

IMPORTANT: Each category must be SELF-CONTAINED and understandable by someone who has NOT seen the original problems.

## Output Format

Return a JSON object with:

```json
{{
    "categories": [
        {{
            "category_name": "Short descriptive name for this type of error",
            "summary": "One sentence describing the core error pattern.",
            "description": "Detailed description of what goes wrong in this category. Be very specific.",
            "example": "A concrete, self-contained example. Format: 'Problem: [simple problem]. Error: [what the model does wrong]. Correct: [what should happen].'",
            "error_type": "Type of error (e.g., {GENERIC_ERROR_TYPES})",
            "why_leads_to_wrong_answer": "Explanation of how this error causes wrong answers"
        }}
    ],
    "failure_assignments": [
        {{
            "failure_id": 1,
            "problem_idx": <problem_idx>,
            "run_id": <run_id>,
            "category_name": "Name of the category this failure belongs to",
            "trace_details": {{
                "trace_specific_location": "Where in the reasoning the error occurred",
                "trace_specific_details": "Specific details about what went wrong"
            }}
        }}
    ]
}}
```
"""


def generate_subsequent_batch_taxonomy_prompt(
    failures: List[FailureRecord],
    existing_categories: List[IssueCategory],
    include_solution: bool = False,
    domain_description: str = "math problems",
    reasoning_label: str = "Model's Reasoning",
    reasoning_source_note: str = ""
) -> str:
    """Generate prompt for analyzing subsequent batches."""
    
    categories_text = "\n\n".join([
        f"""### Category: {cat.category_name}
- Summary: {cat.summary}
- Description: {cat.description}
- Example: {cat.example}
- Error Type: {cat.error_type}
- Why it leads to wrong answer: {cat.why_leads_to_wrong_answer}
- Traces with this issue so far: {cat.trace_count}"""
        for cat in existing_categories
    ])
    
    failures_text = ""
    for i, failure in enumerate(failures, 1):
        failures_text += f"\n## Failure {i}"
        failures_text += format_failure_for_analysis(failure, reasoning_label=reasoning_label)
    
    solution_instruction = ""
    if include_solution:
        solution_instruction = "\nCompare the model's reasoning to the correct solution to identify where it diverged."
    
    return f"""You are an expert at analyzing why language models fail on {domain_description}.

## Existing Issue Categories

{categories_text}

{failures_text}

## Your Task

For each failure:
1. Determine if the error fits one of the EXISTING categories
2. OR create a NEW category if the error is fundamentally different
{solution_instruction}
{reasoning_source_note}

## Output Format

Return a JSON object with:

```json
{{
    "new_categories": [
        {{
            "category_name": "Short descriptive name for NEW error type",
            "summary": "One sentence describing the core error pattern.",
            "description": "Detailed description of what goes wrong.",
            "example": "A concrete example.",
            "error_type": "Type of error (e.g., {GENERIC_ERROR_TYPES})",
            "why_leads_to_wrong_answer": "Explanation of how this error causes wrong answers"
        }}
    ],
    "failure_assignments": [
        {{
            "failure_id": 1,
            "problem_idx": <problem_idx>,
            "run_id": <run_id>,
            "is_new_category": false,
            "category_name": "Name of existing or new category",
            "trace_details": {{
                "trace_specific_location": "Where in the reasoning the error occurred",
                "trace_specific_details": "Specific details about what went wrong"
            }}
        }}
    ]
}}
```

Note: "new_categories" should only contain categories that don't exist yet.
"""


def generate_guidance_prompt(
    selected_categories: List[IssueCategory],
    category_stats: List[CategoryStats],
    total_failures: int,
    prompt_style: str = "detailed",
    sample_traces: Optional[Dict[str, List[str]]] = None,
    domain_description: str = "math problems"
) -> str:
    """Generate prompt for the LLM to create guidance text."""
    
    categories_text = ""
    stats_by_name = {s.category_name: s for s in category_stats}
    
    for i, cat in enumerate(selected_categories, 1):
        stats = stats_by_name.get(cat.category_name)
        failure_count = stats.failure_count if stats else cat.trace_count
        problem_count = stats.problem_count if stats else 0
        coverage_pct = (failure_count / total_failures * 100) if total_failures > 0 else 0
        
        categories_text += f"""
## Category {i}: {cat.category_name}

**Statistics:** {failure_count} failures ({coverage_pct:.1f}%), {problem_count} problems

**Summary:** {cat.summary}

**Description:** {cat.description}

**Example:** {cat.example}

**Error Type:** {cat.error_type}

**Why it leads to wrong answer:** {cat.why_leads_to_wrong_answer}
"""
        
        if sample_traces and cat.category_name in sample_traces:
            traces = sample_traces[cat.category_name]
            if traces:
                categories_text += "\n**Sample Failures:**\n"
                for j, trace in enumerate(traces, 1):
                    categories_text += f"\n--- Sample {j} ---\n{trace}\n"
        
        categories_text += "\n---\n"
    
    style_instruction = ""
    if prompt_style == "short":
        style_instruction = """
Generate SHORT, CONCISE guidance. Each item should be 1-2 sentences.
"""
    else:
        style_instruction = """
Generate DETAILED guidance with examples. Each item should include:
- Description of the error pattern
- Actionable advice on how to avoid it
- WRONG example showing the error
- CORRECT example showing proper approach
"""
    
    return f"""You are an expert at improving language model performance on {domain_description}.

I have identified the following error categories from model failures. Generate guidance to help avoid these errors.

{categories_text}

## Your Task

Generate guidance text that:
1. Addresses each failure category with specific, actionable advice
2. Is written as instructions TO the model
3. Uses concrete examples where helpful
4. Is prioritized by frequency

{style_instruction}

## Critical Constraints

- The goal is ACCURACY, not caution. Never generate guidance that encourages the model to refuse, abstain, or say "not specified" when an answer can be reasonably provided.
- CORRECT examples must always show the model providing a substantive answer. Never show abstention/refusal as the correct behavior.

## Output Format

Return a JSON object with:

```json
{{
    "guidance_items": [
        {{
            "category_name": "Name of the category",
            "guidance_text": "The full guidance text for this category"
        }}
    ],
    "preamble": "1-2 sentence introduction",
    "full_prompt": "Complete enhanced prompt starting with base instruction"
}}
```

The "full_prompt" should start with:
"{BASE_COT_INSTRUCTION}"

Then add your preamble and guidance items.
"""


# =============================================================================
# Category-Level Analysis Functions
# =============================================================================

def compute_category_level_results(
    trace_assignments: List[TraceAssignment],
    selected_category_names: Set[str],
    baseline_detailed: List[Tuple[int, int, bool]],
    taxonomy_detailed: List[Tuple[int, int, bool]],
    num_eval_runs: int
) -> List[CategoryLevelResult]:
    """
    Compute per-category failure rates before and after optimization.
    
    Args:
        trace_assignments: List of trace-to-category assignments
        selected_category_names: Set of category names that were selected for guidance
        baseline_detailed: List of (problem_idx, run_id, is_correct) for baseline
        taxonomy_detailed: List of (problem_idx, run_id, is_correct) for taxonomy
        num_eval_runs: Number of evaluation runs
    
    Returns:
        List of CategoryLevelResult objects
    """
    # Group problems by category
    category_to_problems: Dict[str, Set[int]] = defaultdict(set)
    for assignment in trace_assignments:
        category_to_problems[assignment.category_name].add(assignment.problem_idx)
    
    # Build lookup dictionaries for results
    baseline_dict = {(p, r): c for p, r, c in baseline_detailed}
    taxonomy_dict = {(p, r): c for p, r, c in taxonomy_detailed}
    
    results = []
    
    for category_name, problem_indices in category_to_problems.items():
        problem_list = sorted(list(problem_indices))
        
        baseline_failures = 0
        taxonomy_failures = 0
        total_evals = 0
        
        for prob_idx in problem_list:
            for run_id in range(num_eval_runs):
                key = (prob_idx, run_id)
                total_evals += 1
                
                # Count failures (not correct)
                if key in baseline_dict and not baseline_dict[key]:
                    baseline_failures += 1
                if key in taxonomy_dict and not taxonomy_dict[key]:
                    taxonomy_failures += 1
        
        if total_evals > 0:
            baseline_fail_rate = baseline_failures / total_evals
            taxonomy_fail_rate = taxonomy_failures / total_evals
        else:
            baseline_fail_rate = 0.0
            taxonomy_fail_rate = 0.0
        
        results.append(CategoryLevelResult(
            category_name=category_name,
            problem_indices=problem_list,
            num_problems=len(problem_list),
            baseline_failure_rate=baseline_fail_rate,
            taxonomy_failure_rate=taxonomy_fail_rate,
            delta_failure_rate=taxonomy_fail_rate - baseline_fail_rate,
            was_selected=category_name in selected_category_names,
            baseline_failures=baseline_failures,
            taxonomy_failures=taxonomy_failures,
            total_evals=total_evals
        ))
    
    # Sort by baseline failure rate (descending)
    results.sort(key=lambda x: -x.baseline_failure_rate)
    
    return results


def print_category_level_analysis(results: List[CategoryLevelResult]) -> None:
    """Print category-level analysis results."""
    print(f"\n{'='*80}")
    print("CATEGORY-LEVEL FAILURE ANALYSIS")
    print(f"{'='*80}")
    
    selected = [r for r in results if r.was_selected]
    not_selected = [r for r in results if not r.was_selected]
    
    print(f"\n{'─'*80}")
    print("SELECTED CATEGORIES (received guidance):")
    print(f"{'─'*80}")
    print(f"{'Category':<35} {'#Prob':>6} {'Base FR':>10} {'Tax FR':>10} {'Δ FR':>10}")
    print(f"{'─'*80}")
    
    for r in selected:
        delta_str = f"{r.delta_failure_rate*100:+.1f}%"
        print(f"{r.category_name[:35]:<35} {r.num_problems:>6} "
              f"{r.baseline_failure_rate*100:>9.1f}% {r.taxonomy_failure_rate*100:>9.1f}% {delta_str:>10}")
    
    if selected:
        avg_delta_selected = np.mean([r.delta_failure_rate for r in selected])
        print(f"{'─'*80}")
        print(f"{'AVERAGE (selected)':<35} {'':>6} {'':>10} {'':>10} {avg_delta_selected*100:>+9.1f}%")
    
    if not_selected:
        print(f"\n{'─'*80}")
        print("NOT SELECTED CATEGORIES (no guidance):")
        print(f"{'─'*80}")
        print(f"{'Category':<35} {'#Prob':>6} {'Base FR':>10} {'Tax FR':>10} {'Δ FR':>10}")
        print(f"{'─'*80}")
        
        for r in not_selected:
            delta_str = f"{r.delta_failure_rate*100:+.1f}%"
            print(f"{r.category_name[:35]:<35} {r.num_problems:>6} "
                  f"{r.baseline_failure_rate*100:>9.1f}% {r.taxonomy_failure_rate*100:>9.1f}% {delta_str:>10}")
        
        avg_delta_not_selected = np.mean([r.delta_failure_rate for r in not_selected])
        print(f"{'─'*80}")
        print(f"{'AVERAGE (not selected)':<35} {'':>6} {'':>10} {'':>10} {avg_delta_not_selected*100:>+9.1f}%")
    
    # Summary comparison
    if selected and not_selected:
        avg_selected = np.mean([r.delta_failure_rate for r in selected])
        avg_not_selected = np.mean([r.delta_failure_rate for r in not_selected])
        diff = avg_selected - avg_not_selected
        
        print(f"\n{'='*80}")
        print("SUMMARY: Category Selection Effectiveness")
        print(f"{'='*80}")
        print(f"  Selected categories Δ FR:     {avg_selected*100:+.2f}%")
        print(f"  Not selected categories Δ FR: {avg_not_selected*100:+.2f}%")
        print(f"  Difference:                   {diff*100:+.2f}%")
        
        if avg_selected < avg_not_selected:
            print(f"  ✓ Selected categories show MORE improvement (as expected)")
        else:
            print(f"  ✗ Selected categories do NOT show more improvement")


# =============================================================================
# Taxonomy Serialization
# =============================================================================

def save_taxonomy_to_file(
    filepath: Path,
    issue_categories: List[IssueCategory],
    trace_assignments: List[TraceAssignment],
    failure_records: List[FailureRecord],
    metadata: Dict[str, Any]
) -> None:
    """Save taxonomy to JSON file for later reuse."""
    data = {
        "metadata": metadata,
        "categories": [asdict(c) for c in issue_categories],
        "trace_assignments": [asdict(a) for a in trace_assignments],
        "failure_records": [asdict(f) for f in failure_records]
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"Saved taxonomy to: {filepath}")


def load_taxonomy_from_file(filepath: Path) -> Tuple[
    List[IssueCategory],
    List[TraceAssignment],
    List[FailureRecord],
    Dict[str, Any]
]:
    """Load taxonomy from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    issue_categories = [
        IssueCategory(**c) for c in data.get("categories", [])
    ]
    
    trace_assignments = [
        TraceAssignment(**a) for a in data.get("trace_assignments", [])
    ]
    
    failure_records = [
        FailureRecord(**f) for f in data.get("failure_records", [])
    ]
    
    metadata = data.get("metadata", {})
    
    print(f"Loaded taxonomy from: {filepath}")
    print(f"  Categories: {len(issue_categories)}")
    print(f"  Assignments: {len(trace_assignments)}")
    print(f"  Failure records: {len(failure_records)}")
    
    return issue_categories, trace_assignments, failure_records, metadata


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from model output."""
    text = text.strip()
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:idx + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def get_local_json_response(
    prompt: str,
    *,
    model,
    tokenizer,
    system_message: str = "",
    max_new_tokens: int = 10000,
    do_sample: bool = True,
) -> Dict[str, Any]:
    """Generate and parse a JSON response from a local HF model."""
    raw_text = run_prompt(
        prompt,
        model=model,
        tokenizer=tokenizer,
        system_message=system_message or None,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    response = extract_json_from_text(raw_text)
    if response is None:
        raise ValueError(f"Could not parse JSON from local model output:\n{raw_text[:2000]}")
    return response


def run_re_multi_run_evaluation(
    program: GraphNode,
    dataset: List[REPairExample],
    num_runs: int,
    *,
    model,
    tokenizer,
    batch_size: int,
    yes_token_id: int,
    no_token_id: int,
    log_every: int,
    n_chunks: int = 3,
) -> Tuple[List[float], List[Tuple[int, int, bool]], List[Tuple[REPairExample, str, bool]]]:
    """Run multi-run binary inference evaluation for relation extraction."""
    if num_runs <= 0:
        print(f"\n  Skipping evaluation: num_runs = {num_runs}")
        return [], [], []

    num_problems = len(dataset)
    if num_problems == 0:
        print(f"\n  Skipping evaluation: empty dataset")
        return [], [], []

    total_examples = num_problems * num_runs
    print(f"\n  Expanding {num_problems} problems × {num_runs} runs = {total_examples} evaluations...")

    expanded_dataset = [ex for ex in dataset for _ in range(num_runs)]
    prompts = [
        build_re_inference_prompt(
            node=program,
            relation=example.relation,
            relation_description=example.relation_description,
            support_sentence=example.support_sentence,
            query_sentence=example.query_sentence,
        )
        for example in expanded_dataset
    ]

    predictions = run_binary_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        log_every=log_every,
    )

    detailed_results = []
    trace_rows = []
    per_run_labels: List[List[str]] = [[] for _ in range(num_runs)]
    per_run_predictions: List[List[str]] = [[] for _ in range(num_runs)]

    for idx, (example, prediction) in enumerate(zip(expanded_dataset, predictions)):
        prob_idx = idx // num_runs
        run_id = idx % num_runs
        is_correct = prediction.strip().lower() == example.answer.strip().lower()
        detailed_results.append((prob_idx, run_id, is_correct))
        trace_rows.append((example, prediction, is_correct))
        gold_relation = example.relation if example.answer.strip().lower() == "yes" else NO_RELATION
        predicted_relation = example.relation if prediction.strip().lower() == "yes" else NO_RELATION
        per_run_labels[run_id].append(gold_relation)
        per_run_predictions[run_id].append(predicted_relation)

    per_run_scores = []
    for labels, preds in zip(per_run_labels, per_run_predictions):
        metrics = compute_prf_stats(labels, preds, n_chunks=n_chunks)
        per_run_scores.append(metrics["f1_mean"])

    mean_f1 = sum(per_run_scores) / len(per_run_scores) if per_run_scores else 0.0
    min_f1 = min(per_run_scores) if per_run_scores else 0.0
    max_f1 = max(per_run_scores) if per_run_scores else 0.0
    print(f"  Mean F1: {mean_f1 * 100:.2f}% (min: {min_f1 * 100:.2f}%, max: {max_f1 * 100:.2f}%)")

    return per_run_scores, detailed_results, trace_rows


def run_re_episode_set_evaluation(
    program: GraphNode,
    *,
    method_name: str,
    split_name: str,
    dataset_type: str,
    episodes_data: Dict[str, Any],
    num_runs: int,
    model,
    tokenizer,
    batch_size: int,
    yes_token_id: int,
    no_token_id: int,
    log_every: int,
    n_chunks: int,
    query_index: int,
    output_dir: Optional[Path] = None,
) -> Tuple[List[float], List[Tuple[int, int, bool]]]:
    """Run episode-level RE evaluation using the same evaluator as core_trainer."""
    if num_runs <= 0:
        print(f"\n  Skipping evaluation: num_runs = {num_runs}")
        return [], []

    episodes = episodes_data.get("episodes", [])
    if not episodes:
        print(f"\n  Skipping evaluation: empty {split_name} episodes")
        return [], []

    per_run_scores = []
    eval_output_dir = None
    if output_dir is not None:
        eval_output_dir = str(output_dir / "eval_outputs")
    for run_id in range(num_runs):
        metrics = evaluate_re_episode_fn(
            program,
            split_name,
            dataset_type=dataset_type,
            model=model,
            tokenizer=tokenizer,
            episodes=episodes,
            shots=episodes_data,
            query_index=query_index,
            batch_size=batch_size,
            n_chunks=n_chunks,
            eval_id=f"{method_name}_{split_name}_{run_id}",
            output_dir=eval_output_dir,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            log_every=log_every,
        )
        per_run_scores.append(metrics["f1_mean"])

    mean_f1 = sum(per_run_scores) / len(per_run_scores) if per_run_scores else 0.0
    min_f1 = min(per_run_scores) if per_run_scores else 0.0
    max_f1 = max(per_run_scores) if per_run_scores else 0.0
    print(f"  Mean F1: {mean_f1 * 100:.2f}% (min: {min_f1 * 100:.2f}%, max: {max_f1 * 100:.2f}%)")
    return per_run_scores, []


# =============================================================================
# Main Pipeline Class
# =============================================================================

class UnifiedPromptOptimizer:
    """
    Unified pipeline for comparing prompt optimization methods using DSPy.
    
    V2: Supports factored execution and category-level tracking.
    """
    
    def __init__(
        self,
        # Dataset
        dataset: str = "aimo",
        task_mode: str = "",
        valid_size: int = 90,
        # Methods
        methods: List[str] = None,
        # Models
        main_model: str = DEFAULT_MAIN_MODEL,
        main_temperature: float = DEFAULT_MAIN_TEMPERATURE,
        reflection_model: str = DEFAULT_REFLECTION_MODEL,
        taxonomy_model: str = DEFAULT_TAXONOMY_MODEL,
        # Taxonomy settings
        taxonomy_runs: int = 5,
        taxonomy_batch_size: int = 6,
        include_solution: bool = False,
        coverage_threshold: float = 0.7,
        max_guidances: int = 7,
        min_problems: int = 2,
        prompt_style: str = "detailed",
        include_traces: bool = False,
        num_sample_traces: int = 2,
        # Evaluation settings
        eval_runs: int = 30,
        test_runs: int = 30,
        num_threads: int = 64,
        alpha: float = 0.05,
        # GEPA/MIPROv2 settings
        optimizer_auto: str = "medium",
        train_frac: float = 0.5,
        # Output
        output_dir: str = "unified_optimization_results/",
        experiment_name: Optional[str] = None,
        seed: int = 42,
        verbose: bool = False,
        skip_validation: bool = False,
        # V2: Factored execution
        load_taxonomy: Optional[str] = None,
        save_taxonomy: Optional[str] = None,
        ablation_mode: bool = False,
        # V2: Category-level tracking
        track_category_level: bool = True,
        # V2: Method ablations
        raw_sampling: bool = False,
        num_raw_samples: int = 10,
        direct_categories: bool = False,
        num_guidance_prompts: int = 1,
        # Relation extraction mode
        data_dir: Optional[str] = None,
        train_samples_file: str = "fs_tacred_train_samples.pkl",
        re_inference_mode: str = INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
        re_feedback_prompt_key: str = "mistakes_v1",
        inference_batch_size: int = 8,
        feedback_batch_size: int = 4,
        feedback_max_new_tokens: int = 512,
        log_every: int = 20,
        re_eval_n_chunks: int = 3,
        dev_split: str = "dev",
        test_split: str = "test",
        query_index: int = 0,
        eval_batch_size: int = 8,
        device_map: Optional[str] = None,
        max_new_tokens: Optional[int] = None
    ):
        # Store parameters
        self.dataset = dataset
        self.task_mode = task_mode or ""
        self.is_re_mode = self.task_mode == RE_TASK_MODE
        self.valid_size = valid_size
        self.methods = methods or ["baseline", "taxonomy"]
        self.main_model = main_model
        self.main_temperature = main_temperature
        self.reflection_model = reflection_model
        self.taxonomy_model = taxonomy_model
        self.taxonomy_runs = taxonomy_runs
        self.taxonomy_batch_size = taxonomy_batch_size
        self.include_solution = include_solution
        self.coverage_threshold = coverage_threshold
        self.max_guidances = max_guidances
        self.min_problems = min_problems
        self.prompt_style = prompt_style
        self.include_traces = include_traces
        self.num_sample_traces = num_sample_traces
        self.eval_runs = eval_runs
        self.test_runs = test_runs
        self.num_threads = num_threads
        self.alpha = alpha
        self.optimizer_auto = optimizer_auto
        self.train_frac = train_frac
        self.seed = seed
        self.verbose = verbose
        self.skip_validation = skip_validation
        
        # V2: Factored execution
        self.load_taxonomy_path = Path(load_taxonomy) if load_taxonomy else None
        self.save_taxonomy_path = Path(save_taxonomy) if save_taxonomy else None
        self.ablation_mode = ablation_mode
        self.track_category_level = track_category_level
        
        # V2: Method ablations
        self.raw_sampling = raw_sampling
        self.num_raw_samples = num_raw_samples
        self.direct_categories = direct_categories
        self.num_guidance_prompts = max(1, num_guidance_prompts)
        self.data_dir = data_dir
        self.train_samples_file = train_samples_file
        self.re_inference_mode = re_inference_mode
        self.re_feedback_prompt_key = re_feedback_prompt_key
        self.inference_batch_size = inference_batch_size
        self.feedback_batch_size = feedback_batch_size
        self.feedback_max_new_tokens = feedback_max_new_tokens
        self.log_every = log_every
        self.re_eval_n_chunks = re_eval_n_chunks
        self.dev_split = dev_split
        self.test_split = test_split
        self.query_index = query_index
        self.eval_batch_size = eval_batch_size
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        if self.is_re_mode:
            self.taxonomy_runs = 1
        if self.max_new_tokens is None:
            self.max_new_tokens = 10000 if self.is_re_mode else 32768
        
        # Domain description
        self.domain_description = (
            "relation extraction tasks with binary yes/no inference"
            if self.is_re_mode else get_domain_description(dataset)
        )
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Setup output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.expr_name = experiment_name or f"unified_opt_{dataset}_{timestamp}"
        self.output_dir = Path(output_dir) / self.expr_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Token tracking
        self.token_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
        
        # State
        self.main_lm = None
        self.reflection_lm = None
        self.local_model = None
        self.local_tokenizer = None
        self.taxonomy_local_model = None
        self.taxonomy_local_tokenizer = None
        self.yes_token_id = None
        self.no_token_id = None
        self.feedback_pool: Dict[str, List[dict]] = {}
        self.train_shot_index: Dict[str, dict] = {}
        self.re_dev_data: Dict[str, Any] = {}
        self.re_test_data: Dict[str, Any] = {}
        self.train_set: List[dspy.Example] = []
        self.val_set: List[dspy.Example] = []
        self.test_set: List[dspy.Example] = []
        self.metric = None
        self.metric_with_feedback = None
        self.taxonomy_examples: List[Any] = []
        
        # Programs
        self.programs: Dict[str, Any] = {}
        
        # Results
        self.method_results: Dict[str, Dict[str, MethodResult]] = {}
        
        # Taxonomy state
        self.issue_categories: List[IssueCategory] = []
        self.trace_assignments: List[TraceAssignment] = []
        self.selected_categories: List[IssueCategory] = []
        self.category_stats: List[CategoryStats] = []
        self.generated_prompt: str = ""
        self.generated_prompt_candidates: List[str] = []
        self.failure_records: List[FailureRecord] = []
        
        # V2: Category-level results
        self.category_level_results: List[CategoryLevelResult] = []
    
    def _load_api_key_from_file(self, filepath: str) -> str:
        """Load API key from file."""
        with open(filepath, 'r') as f:
            return f.readline().strip()
    
    def _record_phase_stats(self, phase_name: str, stats: Dict[str, Dict]) -> None:
        """Record and print token stats for a phase."""
        self.token_stats[phase_name] = stats
        print_phase_token_stats(phase_name, stats)

    def _get_taxonomy_generation_backend(self) -> Tuple[Any, Any]:
        """Return model/tokenizer pair for local JSON generation in RE mode."""
        if not self.is_re_mode:
            return None, None
        if self.taxonomy_model == self.main_model:
            return self.local_model, self.local_tokenizer
        return self.taxonomy_local_model, self.taxonomy_local_tokenizer

    def _get_json_response(self, prompt: str, system_message: str) -> Dict[str, Any]:
        """Dispatch JSON generation to the configured backend."""
        if self.is_re_mode:
            model, tokenizer = self._get_taxonomy_generation_backend()
            return get_local_json_response(
                prompt,
                model=model,
                tokenizer=tokenizer,
                system_message=system_message,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
            )
        return get_json_response_from_gpt(
            msg=prompt,
            model=self.taxonomy_model,
            system_message=system_message,
            temperature=0.1
        )

    def _sample_re_examples(self, num_examples: int, rng_seed: int) -> List[REPairExample]:
        """Sample TACRED-style binary inference examples from the train pool."""
        rng = random.Random(rng_seed)
        feedback_samples = sample_feedback_fn(self.feedback_pool, k=num_examples, rng=rng)
        enrich_feedback_samples(feedback_samples, self.train_shot_index)

        examples: List[REPairExample] = []
        for idx, sample in enumerate(feedback_samples.all_samples):
            relation = sample.relation
            examples.append(REPairExample(
                problem_idx=idx,
                relation=relation,
                support_sentence=sample.support_sentence,
                query_sentence=sample.query_sentence,
                answer=sample.label,
                relation_description=get_relation_description(relation, dt=self.dataset),
                id_1shot=str(sample.id_1shot),
                id_query=str(sample.id_query)
            ))
        return examples

    def _setup_relation_extraction_non_reasoning(self):
        """Initialize the TACRED/local-LLM pipeline."""
        print(f"\nConfiguring local main model: {self.main_model}")
        self.local_model, self.local_tokenizer = load_model_and_tokenizer(
            self.main_model,
            device_map=self.device_map,
        )
        self.yes_token_id = _resolve_binary_token_id(self.local_tokenizer, "yes")
        self.no_token_id = _resolve_binary_token_id(self.local_tokenizer, "no")

        if self.taxonomy_model != self.main_model:
            print(f"Configuring local taxonomy model: {self.taxonomy_model}")
            self.taxonomy_local_model, self.taxonomy_local_tokenizer = load_model_and_tokenizer(
                self.taxonomy_model,
                device_map=self.device_map,
            )
        else:
            self.taxonomy_local_model, self.taxonomy_local_tokenizer = self.local_model, self.local_tokenizer

        data_dir = self.data_dir or str(CODES_DIR.parent / "data")
        self.feedback_pool = load_train_samples(data_dir=data_dir, filename=self.train_samples_file)
        self.re_dev_data = load_split_episodes(
            split=self.dev_split,
            data_dir=data_dir,
            dataset_type=self.dataset,
        )
        self.re_test_data = load_split_episodes(
            split=self.test_split,
            data_dir=data_dir,
            dataset_type=self.dataset,
        )
        self.train_shot_index = {}
        for instances in self.feedback_pool.values():
            for inst in instances:
                self.train_shot_index[inst["id"]] = inst

        self.taxonomy_examples = self._sample_re_examples(self.valid_size, self.seed)
        self.val_set = self.re_dev_data.get("episodes", [])
        self.test_set = self.re_test_data.get("episodes", [])
        self.train_set = []
        self.metric = None
        self.metric_with_feedback = None

        feedback_prompt = FEEDBACK_PROMPT_MAP[self.re_feedback_prompt_key]
        self.programs["baseline"] = build_root_node(
            feedback_prompt=feedback_prompt,
            mutation_prompt="",
            inference_mode=self.re_inference_mode,
        )

        print(f"  Taxonomy examples: {len(self.taxonomy_examples)}")
        print(f"  Validation episodes: {len(self.val_set)}")
        print(f"  Test episodes: {len(self.test_set)}")
        print("\nSetup complete.")

    def _clone_re_program_with_instruction(self, instruction_prompt: str) -> GraphNode:
        """Clone the baseline RE program with a revised instruction prompt."""
        base_program = self.programs["baseline"]
        return GraphNode(
            inference_prompt=base_program.inference_prompt,
            inference_mode=base_program.inference_mode,
            inference_instruction_prompt=instruction_prompt,
            inference_answer_instruction_prompt=base_program.inference_answer_instruction_prompt,
            inference_example_prompt=base_program.inference_example_prompt,
            inference_input_prompt=base_program.inference_input_prompt,
            feedback_prompt=base_program.feedback_prompt,
            mutation_prompt=base_program.mutation_prompt,
            example_generation_prompt=base_program.example_generation_prompt,
        )

    def _build_re_failure_problem_text(self, example: REPairExample) -> str:
        """Format an RE failure case for taxonomy analysis."""
        return (
            f"Relation: {example.relation}\n"
            f"Relation Description: {example.relation_description}\n"
            f"Support Instance: {example.support_sentence}\n"
            f"Query: {example.query_sentence}"
        )
    
    def _save_prompt_candidates(self, filename: str) -> None:
        """Persist generated prompt candidates for inspection."""
        if not self.generated_prompt_candidates:
            return
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump({"candidate_prompts": self.generated_prompt_candidates}, f, indent=2)

    def _get_method_program_names(self, method_name: str) -> List[str]:
        """Expand a method name to its registered candidate program names."""
        candidate_prefix = f"{method_name}_"
        candidate_names = sorted(
            key for key in self.programs.keys()
            if key.startswith(candidate_prefix)
        )
        if candidate_names:
            return candidate_names
        return [method_name]

    def _register_taxonomy_candidate_programs(self) -> None:
        """Register one or more taxonomy prompt candidates as evaluable programs."""
        if not self.generated_prompt_candidates:
            return

        # Clear previously registered taxonomy candidate variants.
        keys_to_remove = [
            key for key in self.programs
            if key == "taxonomy" or key.startswith("taxonomy_")
        ]
        for key in keys_to_remove:
            del self.programs[key]

        if self.is_re_mode:
            for idx, prompt_text in enumerate(self.generated_prompt_candidates, start=1):
                self.programs[f"taxonomy_{idx}"] = self._clone_re_program_with_instruction(
                    prompt_text
                )
            return

        for idx, prompt_text in enumerate(self.generated_prompt_candidates, start=1):
            CandidateSignature = create_signature_class(
                prompt_text,
                f"EnhancedReasoningTaskCandidate{idx}"
            )
            self.programs[f"taxonomy_{idx}"] = dspy.ChainOfThought(CandidateSignature)

    def _response_to_full_prompt(
        self,
        response: Dict[str, Any],
        *,
        default_preamble: str = "Check your work against these common errors:"
    ) -> str:
        """Convert a single guidance-generation response to a prompt string."""
        prompt_text = response.get("full_prompt", "")
        if prompt_text:
            return prompt_text.strip()

        revised = response.get("revised_instruction_prompt", "")
        if revised:
            return revised.strip()

        preamble = response.get("preamble", default_preamble)
        items = response.get("guidance_items", [])
        items_text = "\n\n".join([f"• {item.get('guidance_text', '')}" for item in items])
        return f"{BASE_COT_INSTRUCTION}\n\n{preamble}\n\n{items_text}".strip()
    
    def setup(self):
        """Initialize LMs, load data, create base program."""
        print(f"\n{'='*70}")
        print("SETUP")
        print(f"{'='*70}")

        if self.is_re_mode:
            self._setup_relation_extraction_non_reasoning()
            return
        
        # Load API keys
        main_api_key = self._load_api_key_from_file(MAIN_API_KEY_FILE)
        print(f"Loaded main API key from: {MAIN_API_KEY_FILE}")
        
        # Configure main LM
        print(f"\nConfiguring main LM: {self.main_model}")
        self.main_lm = dspy.LM(
            self.main_model,
            temperature=self.main_temperature,
            api_key=main_api_key,
            max_tokens=DEFAULT_MAIN_MAX_TOKENS,
            cache=False
        )
        dspy.configure(lm=self.main_lm)
        
        # Configure reflection LM for GEPA
        if "gepa" in self.methods:
            reflection_api_key = self._load_api_key_from_file(REFLECTION_API_KEY_FILE)
            print(f"Loaded reflection API key from: {REFLECTION_API_KEY_FILE}")
            
            print(f"Configuring reflection LM: {self.reflection_model}")
            self.reflection_lm = dspy.LM(
                model=self.reflection_model,
                temperature=DEFAULT_REFLECTION_TEMPERATURE,
                api_key=reflection_api_key,
                max_tokens=DEFAULT_REFLECTION_MAX_TOKENS,
                cache=False
            )
        
        # Load dataset
        print(f"\nLoading dataset: {self.dataset}")
        validation_examples = get_validation_examples(self.dataset, self.valid_size)
        test_examples = get_test_examples(self.dataset)
        
        print(f"  Validation examples: {len(validation_examples)}")
        print(f"  Test examples: {len(test_examples)}")
        
        # Convert to DSPy format
        all_valid = [
            dspy.Example(
                problem=x['problem'],
                answer=str(x['answer']),
                solution=x.get('solution', '')
            ).with_inputs("problem")
            for x in validation_examples
        ]
        
        self.test_set = [
            dspy.Example(
                problem=x['problem'],
                answer=str(x['answer'])
            ).with_inputs("problem")
            for x in test_examples
        ]
        
        # Shuffle and split validation for GEPA/MIPROv2
        random.Random(self.seed).shuffle(all_valid)
        split_idx = int(self.train_frac * len(all_valid))
        self.train_set = all_valid[:split_idx]
        self.val_set = all_valid[split_idx:]
        
        # For taxonomy, use all validation examples
        self.taxonomy_examples = all_valid.copy()
        
        print(f"  Train set (for GEPA/MIPROv2): {len(self.train_set)}")
        print(f"  Val set (for GEPA/MIPROv2): {len(self.val_set)}")
        print(f"  Taxonomy examples: {len(self.taxonomy_examples)}")
        print(f"  Test set: {len(self.test_set)}")
        
        # Create metrics
        self.metric = create_metric(self.dataset)
        self.metric_with_feedback = create_metric_with_feedback(self.dataset)
        
        # Create base signature and program
        BaseSignature = create_signature_class(BASE_COT_INSTRUCTION, "BaseReasoningTask")
        self.programs["baseline"] = dspy.ChainOfThought(BaseSignature)
        
        print("\nSetup complete.")
    
    def run_baseline_for_taxonomy(self) -> List[FailureRecord]:
        """Run baseline evaluation to collect failures for taxonomy."""
        print(f"\n{'='*70}")
        print("TAXONOMY: COLLECTING FAILURES FROM BASELINE")
        print(f"{'='*70}")

        if self.is_re_mode:
            if self.taxonomy_runs <= 0:
                print(f"\n  Skipping baseline for taxonomy: taxonomy_runs = {self.taxonomy_runs}")
                self.failure_records = []
                self._record_phase_stats("baseline_for_taxonomy", {})
                return []

            per_run_f1_scores, _, trace_rows = run_re_multi_run_evaluation(
                program=self.programs["baseline"],
                dataset=self.taxonomy_examples,
                num_runs=self.taxonomy_runs,
                model=self.local_model,
                tokenizer=self.local_tokenizer,
                batch_size=self.inference_batch_size,
                yes_token_id=self.yes_token_id,
                no_token_id=self.no_token_id,
                log_every=self.log_every,
                n_chunks=self.re_eval_n_chunks,
            )

            failed_samples = FeedbackSamples()
            failure_records = []
            for idx, (example, prediction, is_correct) in enumerate(trace_rows):
                if is_correct:
                    continue
                run_id = idx % self.taxonomy_runs
                failed_sample = FeedbackSample(
                    id_1shot=example.id_1shot,
                    id_query=example.id_query,
                    inference=prediction,
                    label=example.answer,
                )
                failed_sample.relation = example.relation
                failed_sample.support_sentence = example.support_sentence
                failed_sample.query_sentence = example.query_sentence
                failed_samples.add_to_all_samples(failed_sample)
                failed_samples.add_to_selected_samples(failed_sample)

            if failed_samples.selected_samples:
                failed_samples = generate_re_feedback_fn(
                    self.programs["baseline"],
                    failed_samples,
                    dataset_type=self.dataset,
                    model=self.local_model,
                    tokenizer=self.local_tokenizer,
                    batch_size=self.feedback_batch_size,
                    max_new_tokens=self.feedback_max_new_tokens,
                    do_sample=True,
                )

            feedback_by_key = {}
            for sample in failed_samples.selected_samples:
                feedback_by_key[(str(sample.id_1shot), str(sample.id_query), sample.inference)] = sample.feedback_text

            for idx, (example, prediction, is_correct) in enumerate(trace_rows):
                if is_correct:
                    continue
                run_id = idx % self.taxonomy_runs
                failure_records.append(FailureRecord(
                    problem_idx=example.problem_idx,
                    run_id=run_id,
                    problem=self._build_re_failure_problem_text(example),
                    correct_answer=example.answer,
                    predicted_answer=prediction,
                    reasoning=feedback_by_key.get(
                        (example.id_1shot, example.id_query, prediction),
                        "No feedback generated."
                    ),
                    solution=None
                ))

            overall_f1 = sum(per_run_f1_scores) / len(per_run_f1_scores) if per_run_f1_scores else 0.0
            print(f"\n  Collected {len(failure_records)} mistakes from {len(set(f.problem_idx for f in failure_records))} problems")
            print(f"  F1 from scorer: {overall_f1 * 100:.2f}%")
            self.failure_records = failure_records
            self._record_phase_stats("baseline_for_taxonomy", {})
            return failure_records
        
        clear_lm_history(self.main_lm, "before baseline_for_taxonomy")
        
        if self.taxonomy_runs <= 0:
            print(f"\n  Skipping baseline for taxonomy: taxonomy_runs = {self.taxonomy_runs}")
            self.failure_records = []
            phase_stats = convert_dspy_history_to_phase_stats(self.main_lm, self.main_model)
            self._record_phase_stats("baseline_for_taxonomy", phase_stats)
            return []
        
        num_problems = len(self.taxonomy_examples)
        
        if num_problems == 0:
            print(f"\n  Skipping baseline for taxonomy: empty dataset")
            self.failure_records = []
            phase_stats = convert_dspy_history_to_phase_stats(self.main_lm, self.main_model)
            self._record_phase_stats("baseline_for_taxonomy", phase_stats)
            return []
        
        num_runs = self.taxonomy_runs
        total_examples = num_problems * num_runs
        
        print(f"\n  Expanding {num_problems} problems × {num_runs} runs = {total_examples} evaluations...")
        
        expanded_dataset = [ex for ex in self.taxonomy_examples for _ in range(num_runs)]
        
        evaluator = dspy.Evaluate(
            devset=expanded_dataset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            display_table=True,
            provide_traceback=True,
        )
        
        eval_result = evaluator(self.programs["baseline"])
        
        failures = []
        total_correct = 0
        
        for idx, (example, prediction, score) in enumerate(eval_result.results):
            prob_idx = idx // num_runs
            run_id = idx % num_runs
            
            if score > 0:
                total_correct += 1
            else:
                reasoning = getattr(prediction, 'reasoning', '') or ''
                pred_answer = getattr(prediction, 'answer', '') or str(prediction)
                solution = getattr(example, 'solution', '') if self.include_solution else None
                
                failures.append(FailureRecord(
                    problem_idx=prob_idx,
                    run_id=run_id,
                    problem=str(example.problem),
                    correct_answer=str(example.answer),
                    predicted_answer=pred_answer,
                    reasoning=reasoning,
                    solution=solution
                ))
        
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        print(f"\n  Overall accuracy: {overall_accuracy * 100:.2f}% ({total_correct}/{total_examples})")
        
        self.failure_records = failures
        num_problems_with_failures = len(set(f.problem_idx for f in failures))
        print(f"  Collected {len(failures)} failures from {num_problems_with_failures} problems")
        
        phase_stats = convert_dspy_history_to_phase_stats(self.main_lm, self.main_model)
        self._record_phase_stats("baseline_for_taxonomy", phase_stats)
        
        return failures
    
    def build_taxonomy(self, failures: List[FailureRecord]):
        """Build failure taxonomy from collected failures."""
        print(f"\n{'='*70}")
        print("TAXONOMY: BUILDING FAILURE TAXONOMY")
        print(f"{'='*70}")
        
        snapshot_before = capture_token_snapshot()
        
        if not failures:
            print("No failures to analyze!")
            snapshot_after = capture_token_snapshot()
            delta = compute_phase_delta(snapshot_before, snapshot_after)
            self._record_phase_stats("taxonomy_building", delta)
            return
        
        shuffled = failures.copy()
        random.shuffle(shuffled)
        
        num_batches = (len(shuffled) + self.taxonomy_batch_size - 1) // self.taxonomy_batch_size
        print(f"\nProcessing {len(shuffled)} failures in {num_batches} batches")
        taxonomy_start = time.monotonic()
        
        for batch_idx in range(num_batches):
            start = batch_idx * self.taxonomy_batch_size
            end = min(start + self.taxonomy_batch_size, len(shuffled))
            batch = shuffled[start:end]
            processed = end
            elapsed = time.monotonic() - taxonomy_start
            
            print(f"\n--- Batch {batch_idx + 1}/{num_batches}: {len(batch)} failures ---")
            print(
                f"  Progress: processed {processed}/{len(shuffled)} failures "
                f"(elapsed={elapsed:.2f}s)"
            )
            
            if batch_idx == 0:
                prompt = generate_first_batch_taxonomy_prompt(
                    batch,
                    include_solution=self.include_solution,
                    domain_description=self.domain_description,
                    reasoning_label=(
                        "Most Likely Reasoning Behind The Incorrect Binary Prediction"
                        if self.is_re_mode else "Model's Reasoning"
                    ),
                    reasoning_source_note=(
                        "Important: the reasoning field is post-hoc feedback describing the most likely cause of the incorrect binary prediction, not a verbatim chain-of-thought from the original model. Use it as probabilistic evidence together with the input, gold label, and wrong answer."
                        if self.is_re_mode else ""
                    )
                )
            else:
                prompt = generate_subsequent_batch_taxonomy_prompt(
                    batch,
                    existing_categories=self.issue_categories,
                    include_solution=self.include_solution,
                    domain_description=self.domain_description,
                    reasoning_label=(
                        "Most Likely Reasoning Behind The Incorrect Binary Prediction"
                        if self.is_re_mode else "Model's Reasoning"
                    ),
                    reasoning_source_note=(
                        "Important: the reasoning field is post-hoc feedback describing the most likely cause of the incorrect binary prediction, not a verbatim chain-of-thought from the original model. Use it as probabilistic evidence together with the input, gold label, and wrong answer."
                        if self.is_re_mode else ""
                    )
                )
            
            if self.verbose:
                print(f"\nTaxonomy Prompt:\n{prompt}")
            
            print("  Analyzing failures with LLM...")
            response = self._get_json_response(
                prompt,
                "You are an expert at analyzing language model failures. Return valid JSON."
            )
            
            if not response:
                print("  Warning: Empty response from LLM")
                continue
            
            if batch_idx == 0:
                categories_data = response.get("categories", [])
                for cat_data in categories_data:
                    self.issue_categories.append(IssueCategory(
                        category_name=cat_data.get("category_name", "Unknown"),
                        summary=cat_data.get("summary", ""),
                        description=cat_data.get("description", ""),
                        example=cat_data.get("example", ""),
                        error_type=cat_data.get("error_type", ""),
                        why_leads_to_wrong_answer=cat_data.get("why_leads_to_wrong_answer", ""),
                        trace_count=0
                    ))
                print(f"  Created {len(categories_data)} categories")
            else:
                new_categories = response.get("new_categories", [])
                for cat_data in new_categories:
                    self.issue_categories.append(IssueCategory(
                        category_name=cat_data.get("category_name", "Unknown"),
                        summary=cat_data.get("summary", ""),
                        description=cat_data.get("description", ""),
                        example=cat_data.get("example", ""),
                        error_type=cat_data.get("error_type", ""),
                        why_leads_to_wrong_answer=cat_data.get("why_leads_to_wrong_answer", ""),
                        trace_count=0
                    ))
                if new_categories:
                    print(f"  Created {len(new_categories)} new categories")
            
            assignments = response.get("failure_assignments", [])
            for assignment in assignments:
                cat_name = assignment.get("category_name", "Unknown")
                trace_details = assignment.get("trace_details", {})
                
                self.trace_assignments.append(TraceAssignment(
                    problem_idx=assignment.get("problem_idx", 0),
                    run_id=assignment.get("run_id", 0),
                    category_name=cat_name,
                    trace_specific_location=trace_details.get("trace_specific_location", ""),
                    trace_specific_details=trace_details.get("trace_specific_details", "")
                ))
                
                for cat in self.issue_categories:
                    if cat.category_name == cat_name:
                        cat.trace_count += 1
                        break
            
            print(f"  Assigned {len(assignments)} failures")
            print(f"  Total categories: {len(self.issue_categories)}")

        taxonomy_elapsed = time.monotonic() - taxonomy_start
        print(f"\nCompleted taxonomy analysis of {len(shuffled)} failures in {taxonomy_elapsed:.2f}s")
        
        # Save taxonomy
        taxonomy_file = self.output_dir / "taxonomy.json"
        with open(taxonomy_file, 'w') as f:
            json.dump({
                "categories": [asdict(c) for c in self.issue_categories],
                "assignments": [asdict(a) for a in self.trace_assignments],
                "total_failures": len(failures)
            }, f, indent=2, cls=NumpyEncoder)
        print(f"\nTaxonomy saved to: {taxonomy_file}")
        
        # V2: Save full taxonomy for reuse if requested
        if self.save_taxonomy_path:
            save_taxonomy_to_file(
                self.save_taxonomy_path,
                self.issue_categories,
                self.trace_assignments,
                self.failure_records,
                metadata={
                    "dataset": self.dataset,
                    "valid_size": self.valid_size,
                    "taxonomy_runs": self.taxonomy_runs,
                    "include_solution": self.include_solution,
                    "seed": self.seed
                }
            )
        
        token_tracker.aggregate_thread_costs()
        snapshot_after = capture_token_snapshot()
        delta = compute_phase_delta(snapshot_before, snapshot_after)
        self._record_phase_stats("taxonomy_building", delta)
    
    def select_categories(self):
        """Select top categories based on coverage threshold."""
        print(f"\n{'='*70}")
        print("TAXONOMY: SELECTING TOP CATEGORIES")
        print(f"{'='*70}")
        
        if not self.issue_categories:
            print("No categories to select!")
            return
        
        category_to_problems: Dict[str, Set[int]] = defaultdict(set)
        for assignment in self.trace_assignments:
            category_to_problems[assignment.category_name].add(assignment.problem_idx)
        
        total_failures = len(self.trace_assignments)
        
        seen_names = set()
        self.category_stats = []
        for cat in self.issue_categories:
            if cat.category_name in seen_names:
                continue
            seen_names.add(cat.category_name)
            
            problems = category_to_problems.get(cat.category_name, set())
            self.category_stats.append(CategoryStats(
                category_name=cat.category_name,
                failure_count=cat.trace_count,
                problem_count=len(problems),
                problem_list=sorted(list(problems)),
                coverage_fraction=cat.trace_count / total_failures if total_failures > 0 else 0
            ))
        
        self.category_stats.sort(key=lambda s: -s.failure_count)
        
        eligible = [s for s in self.category_stats if s.problem_count >= self.min_problems]
        
        selected_stats = []
        cumulative_coverage = 0.0
        
        print(f"\nSelection criteria:")
        print(f"  Coverage threshold: {self.coverage_threshold*100:.0f}%")
        print(f"  Max guidances: {self.max_guidances}")
        print(f"  Min problems: {self.min_problems}")
        
        for stats in eligible:
            if len(selected_stats) >= self.max_guidances:
                break
            if cumulative_coverage >= self.coverage_threshold:
                break
            
            selected_stats.append(stats)
            cumulative_coverage += stats.coverage_fraction
            
            print(f"  {len(selected_stats)}. {stats.category_name}")
            print(f"      Failures: {stats.failure_count} ({stats.coverage_fraction*100:.1f}%)")
            print(f"      Cumulative: {cumulative_coverage*100:.1f}%")
        
        selected_names = {s.category_name for s in selected_stats}
        self.selected_categories = []
        seen_selected = set()
        for cat in self.issue_categories:
            if cat.category_name in selected_names and cat.category_name not in seen_selected:
                seen_selected.add(cat.category_name)
                self.selected_categories.append(cat)
        
        print(f"\nSelected {len(self.selected_categories)} categories covering {cumulative_coverage*100:.1f}%")
    
    def generate_guidance(self):
        """Generate guidance prompt via LLM."""
        print(f"\n{'='*70}")
        print("TAXONOMY: GENERATING GUIDANCE PROMPT")
        print(f"{'='*70}")
        
        snapshot_before = capture_token_snapshot()
        
        if not self.selected_categories:
            print("No categories selected!")
            snapshot_after = capture_token_snapshot()
            delta = compute_phase_delta(snapshot_before, snapshot_after)
            self._record_phase_stats("guidance_generation", delta)
            return

        if self.is_re_mode:
            selected_stats = [
                s for s in self.category_stats
                if s.category_name in {c.category_name for c in self.selected_categories}
            ]
            categories_text = ""
            for i, cat in enumerate(self.selected_categories, 1):
                stats = next((s for s in selected_stats if s.category_name == cat.category_name), None)
                failure_count = stats.failure_count if stats else cat.trace_count
                problem_count = stats.problem_count if stats else 0
                coverage_pct = (failure_count / len(self.trace_assignments) * 100) if self.trace_assignments else 0
                categories_text += f"""
## Category {i}: {cat.category_name}

**Statistics:** {failure_count} failures ({coverage_pct:.1f}%), {problem_count} problems

**Summary:** {cat.summary}

**Description:** {cat.description}

**Example:** {cat.example}

**Error Type:** {cat.error_type}

**Why it leads to wrong answer:** {cat.why_leads_to_wrong_answer}
"""

            base_instruction = self.programs["baseline"].inference_instruction_prompt
            prompt = f"""You are an expert at improving language model performance on relation extraction with binary yes/no inference.

I have identified the following recurring error categories from model failures. Use them to revise the current instruction prompt so the model can avoid these mistakes.

## Current Instruction Prompt

```text
{base_instruction}
```

## Error Categories

```text
{categories_text}
```

## Your Task

Generate a revised instruction prompt that:
1. Addresses each failure category with specific, actionable advice
2. Improves and generalizes binary relation extraction decisions

## Critical Constraints

- Do not revise the instruction in a way that requires step-by-step reasoning or explanatory output; the task should remain direct binary yes/no inference.
- Preserve compatibility with the existing prompt structure where the answer instruction and input template are appended separately.
- The goal is ACCURACY, not caution. Never generate guidance that encourages the model to refuse, abstain, or say "not specified" when an answer can be provided.

## Output Format

Return a JSON object with:
{{
  "revised_instruction_prompt": "the revised instruction prompt text"
}}
"""
            print("Asking LLM to generate a revised instruction prompt...")
            self.generated_prompt_candidates = []
            for _ in range(self.num_guidance_prompts):
                response = self._get_json_response(
                    prompt,
                    "You are an expert at improving relation extraction prompts. Return valid JSON."
                )
                candidate_prompt = response.get("revised_instruction_prompt", "").strip()
                if not candidate_prompt:
                    raise ValueError("Missing 'revised_instruction_prompt' in RE guidance generation response")
                self.generated_prompt_candidates.append(candidate_prompt)
            self.generated_prompt = self.generated_prompt_candidates[0]

            prompt_file = self.output_dir / "generated_prompt.txt"
            with open(prompt_file, 'w') as f:
                f.write(self.generated_prompt)
            self._save_prompt_candidates("generated_prompt_candidates.json")

            print(f"\nGenerated prompt ({len(self.generated_prompt)} chars)")
            print(f"Saved to: {prompt_file}")
            if len(self.generated_prompt_candidates) > 1:
                print(f"Saved {len(self.generated_prompt_candidates)} prompt candidates")
            print(f"\nPreview:\n{self.generated_prompt}")

            self._register_taxonomy_candidate_programs()
            self._record_phase_stats("guidance_generation", {})
            return
        
        sample_traces = None
        if self.include_traces and self.failure_records:
            sample_traces = {}
            category_failures: Dict[str, List[FailureRecord]] = defaultdict(list)
            
            for assignment in self.trace_assignments:
                for failure in self.failure_records:
                    if (failure.problem_idx == assignment.problem_idx and
                        failure.run_id == assignment.run_id):
                        category_failures[assignment.category_name].append(failure)
                        break
            
            for cat in self.selected_categories:
                failures = category_failures.get(cat.category_name, [])
                traces = []
                for f in failures[:self.num_sample_traces]:
                    trace_text = f"Problem: {f.problem}\nReasoning: {f.reasoning}\nAnswer: {f.predicted_answer}"
                    traces.append(trace_text)
                sample_traces[cat.category_name] = traces
        
        selected_stats = [
            s for s in self.category_stats
            if s.category_name in {c.category_name for c in self.selected_categories}
        ]
        
        prompt = generate_guidance_prompt(
            selected_categories=self.selected_categories,
            category_stats=selected_stats,
            total_failures=len(self.trace_assignments),
            prompt_style=self.prompt_style,
            sample_traces=sample_traces,
            domain_description=self.domain_description
        )
        
        print("Asking LLM to generate guidance...")
        self.generated_prompt_candidates = []
        for _ in range(self.num_guidance_prompts):
            response = self._get_json_response(
                prompt,
                "You are an expert at improving language model performance. Return valid JSON."
            )
            if not response:
                continue
            self.generated_prompt_candidates.append(self._response_to_full_prompt(response))

        if not self.generated_prompt_candidates:
            print("Warning: Empty response")
            token_tracker.aggregate_thread_costs()
            snapshot_after = capture_token_snapshot()
            delta = compute_phase_delta(snapshot_before, snapshot_after)
            self._record_phase_stats("guidance_generation", delta)
            return
        self.generated_prompt = self.generated_prompt_candidates[0]
        
        prompt_file = self.output_dir / "generated_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(self.generated_prompt)
        self._save_prompt_candidates("generated_prompt_candidates.json")
        
        print(f"\nGenerated prompt ({len(self.generated_prompt)} chars)")
        print(f"Saved to: {prompt_file}")
        if len(self.generated_prompt_candidates) > 1:
            print(f"Saved {len(self.generated_prompt_candidates)} prompt candidates")
        print(f"\nPreview:\n{self.generated_prompt}")
        
        self._register_taxonomy_candidate_programs()
        
        token_tracker.aggregate_thread_costs()
        snapshot_after = capture_token_snapshot()
        delta = compute_phase_delta(snapshot_before, snapshot_after)
        self._record_phase_stats("guidance_generation", delta)
    
    def generate_guidance_from_raw_samples(self, failures: List[FailureRecord]):
        """
        Ablation 1: Generate guidance from raw failure samples, skipping taxonomy.
        
        Samples up to num_raw_samples failures (1 per problem max) and passes
        them directly to the guidance generator.
        """
        print(f"\n{'='*70}")
        print("ABLATION: GENERATING GUIDANCE FROM RAW FAILURE SAMPLES")
        print(f"{'='*70}")
        
        snapshot_before = capture_token_snapshot()
        
        if not failures:
            print("No failures to sample!")
            snapshot_after = capture_token_snapshot()
            delta = compute_phase_delta(snapshot_before, snapshot_after)
            self._record_phase_stats("guidance_generation", delta)
            return
        
        # Sample 1 failure per problem (for diversity), up to num_raw_samples
        problem_to_failures: Dict[int, List[FailureRecord]] = defaultdict(list)
        for f in failures:
            problem_to_failures[f.problem_idx].append(f)
        
        sampled_failures = []
        problem_indices = list(problem_to_failures.keys())
        random.shuffle(problem_indices)
        
        for prob_idx in problem_indices[:self.num_raw_samples]:
            # Take first failure from this problem
            sampled_failures.append(problem_to_failures[prob_idx][0])
        
        print(f"Sampled {len(sampled_failures)} failures from {len(problem_indices)} problems")
        
        # Format raw traces
        traces_text = ""
        for i, f in enumerate(sampled_failures, 1):
            traces_text += f"""
--- Failure {i} ---
Problem: {f.problem}

Model's Reasoning:
{f.reasoning}

Model's Answer: {f.predicted_answer}
Correct Answer: {f.correct_answer}

"""
        
        # Generate prompt for guidance generation from raw traces
        prompt = f"""You are an expert at improving language model performance on {self.domain_description}.

I have collected the following failure traces from a model. Analyze these failures and generate guidance to help avoid similar errors.

{traces_text}

## Your Task

Analyze these failures and generate guidance that:
1. Identifies common error patterns across failures
2. Provides specific, actionable advice to avoid these errors
3. Is written as instructions TO the model
4. Uses concrete examples where helpful

{"Generate SHORT, CONCISE guidance. Each item should be 1-2 sentences." if self.prompt_style == "short" else "Generate DETAILED guidance with examples showing wrong vs correct approaches."}

## Critical Constraints

- The goal is ACCURACY, not caution. Never generate guidance that encourages the model to refuse, abstain, or say "not specified" when an answer can be reasonably provided.
- CORRECT examples must always show the model providing a substantive answer.

## Output Format

Return a JSON object with:

```json
{{
    "guidance_items": [
        {{
            "error_pattern": "Description of the error pattern",
            "guidance_text": "The full guidance text"
        }}
    ],
    "preamble": "1-2 sentence introduction",
    "full_prompt": "Complete enhanced prompt starting with base instruction"
}}
```

The "full_prompt" should start with:
"{BASE_COT_INSTRUCTION}"

Then add your preamble and guidance items.
"""
        
        print("Asking LLM to generate guidance from raw traces...")
        self.generated_prompt_candidates = []
        for _ in range(self.num_guidance_prompts):
            response = self._get_json_response(
                prompt,
                "You are an expert at improving language model performance. Return valid JSON."
            )
            if not response:
                continue
            self.generated_prompt_candidates.append(self._response_to_full_prompt(response))

        if not self.generated_prompt_candidates:
            print("Warning: Empty response")
            token_tracker.aggregate_thread_costs()
            snapshot_after = capture_token_snapshot()
            delta = compute_phase_delta(snapshot_before, snapshot_after)
            self._record_phase_stats("guidance_generation", delta)
            return
        self.generated_prompt = self.generated_prompt_candidates[0]
        
        prompt_file = self.output_dir / "generated_prompt_raw_sampling.txt"
        with open(prompt_file, 'w') as f:
            f.write(self.generated_prompt)
        self._save_prompt_candidates("generated_prompt_raw_sampling_candidates.json")
        
        print(f"\nGenerated prompt ({len(self.generated_prompt)} chars)")
        print(f"Saved to: {prompt_file}")
        if len(self.generated_prompt_candidates) > 1:
            print(f"Saved {len(self.generated_prompt_candidates)} prompt candidates")
        print(f"\nPreview:\n{self.generated_prompt}")
        
        self._register_taxonomy_candidate_programs()
        
        token_tracker.aggregate_thread_costs()
        snapshot_after = capture_token_snapshot()
        delta = compute_phase_delta(snapshot_before, snapshot_after)
        self._record_phase_stats("guidance_generation", delta)
    
    def generate_direct_category_prompt(self):
        """
        Ablation 2: Generate prompt by directly inserting category descriptions.
        
        Skips the LLM guidance generation step and formats category information
        directly into the prompt.
        """
        print(f"\n{'='*70}")
        print("ABLATION: DIRECT CATEGORY INSERTION (SKIP GUIDANCE GENERATION)")
        print(f"{'='*70}")
        
        snapshot_before = capture_token_snapshot()
        
        if not self.selected_categories:
            print("No categories selected!")
            snapshot_after = capture_token_snapshot()
            delta = compute_phase_delta(snapshot_before, snapshot_after)
            self._record_phase_stats("guidance_generation", delta)
            return
        
        # Format categories directly as bullet list
        categories_text = "Watch out for these common error patterns:\n\n"
        
        for cat in self.selected_categories:
            categories_text += f"""• {cat.category_name}
  - Summary: {cat.summary}
  - Description: {cat.description}
  - Error Type: {cat.error_type}
  - Why it fails: {cat.why_leads_to_wrong_answer}
  - Example: {cat.example}

"""
        
        self.generated_prompt = f"{BASE_COT_INSTRUCTION}\n\n{categories_text}"
        
        prompt_file = self.output_dir / "generated_prompt_direct_categories.txt"
        with open(prompt_file, 'w') as f:
            f.write(self.generated_prompt)
        
        print(f"\nGenerated prompt ({len(self.generated_prompt)} chars)")
        print(f"Saved to: {prompt_file}")
        print(f"\nPreview:\n{self.generated_prompt[:2000]}...")
        
        EnhancedSignature = create_signature_class(self.generated_prompt, "EnhancedReasoningTask")
        self.programs["taxonomy"] = dspy.ChainOfThought(EnhancedSignature)
        
        # No LLM call, so minimal token usage
        snapshot_after = capture_token_snapshot()
        delta = compute_phase_delta(snapshot_before, snapshot_after)
        self._record_phase_stats("guidance_generation", delta)
    
    def run_gepa_optimization(self):
        """Run GEPA optimization."""
        if self.is_re_mode:
            print("\nSkipping GEPA in relation_extraction_non_reasoning mode")
            return
        print(f"\n{'='*70}")
        print("GEPA OPTIMIZATION")
        print(f"{'='*70}")
        
        clear_lm_history(self.main_lm, "before GEPA")
        if self.reflection_lm:
            clear_lm_history(self.reflection_lm, "before GEPA")
        
        BaseSignature = create_signature_class(BASE_COT_INSTRUCTION, "GEPABaseTask")
        base_program = dspy.ChainOfThought(BaseSignature)
        
        optimizer = GEPA(
            metric=self.metric_with_feedback,
            auto=self.optimizer_auto,
            num_threads=self.num_threads,
            track_stats=True,
            reflection_minibatch_size=3,
            reflection_lm=self.reflection_lm
        )
        
        print(f"\nOptimizing with GEPA (auto={self.optimizer_auto})...")
        optimized = optimizer.compile(
            base_program,
            trainset=self.train_set,
            valset=self.val_set
        )
        
        self.programs["gepa"] = optimized
        
        main_stats = convert_dspy_history_to_phase_stats(self.main_lm, self.main_model)
        if self.reflection_lm:
            reflection_stats = convert_dspy_history_to_phase_stats(self.reflection_lm, self.reflection_model)
            main_stats.update(reflection_stats)
        
        self._record_phase_stats("gepa_optimization", main_stats)
        
        try:
            instruction = optimized.predict.signature.instructions
            print(f"\nGEPA Optimized Instruction:\n{instruction}")
            
            gepa_prompt_file = self.output_dir / "gepa_prompt.txt"
            with open(gepa_prompt_file, 'w') as f:
                f.write(instruction)
        except Exception as e:
            print(f"Could not extract GEPA instruction: {e}")
    
    def run_mipro_optimization(self):
        """Run MIPROv2 optimization."""
        if self.is_re_mode:
            print("\nSkipping MIPROv2 in relation_extraction_non_reasoning mode")
            return
        print(f"\n{'='*70}")
        print("MIPROv2 OPTIMIZATION")
        print(f"{'='*70}")
        
        clear_lm_history(self.main_lm, "before MIPROv2")
        
        BaseSignature = create_signature_class(BASE_COT_INSTRUCTION, "MIPROBaseTask")
        base_program = dspy.ChainOfThought(BaseSignature)
        
        optimizer = MIPROv2(
            metric=self.metric,
            auto=self.optimizer_auto,
            num_threads=self.num_threads,
            track_stats=True
        )
        
        print(f"\nOptimizing with MIPROv2 (auto={self.optimizer_auto})...")
        optimized = optimizer.compile(
            base_program,
            trainset=self.train_set + self.val_set
        )
        
        self.programs["mipro"] = optimized
        
        phase_stats = convert_dspy_history_to_phase_stats(self.main_lm, self.main_model)
        self._record_phase_stats("mipro_optimization", phase_stats)
        
        try:
            instruction = optimized.predict.signature.instructions
            print(f"\nMIPROv2 Optimized Instruction:\n{instruction}")
            
            mipro_prompt_file = self.output_dir / "mipro_prompt.txt"
            with open(mipro_prompt_file, 'w') as f:
                f.write(instruction)
        except Exception as e:
            print(f"Could not extract MIPROv2 instruction: {e}")
    
    def evaluate_all_methods(self):
        """Evaluate all methods on validation and test sets."""
        print(f"\n{'='*70}")
        print("FINAL EVALUATION")
        print(f"{'='*70}")

        skip_test_evaluation = True

        methods_to_evaluate = []
        seen_method_names = set()
        for method_name in self.methods:
            for name in self._get_method_program_names(method_name):
                if name not in seen_method_names:
                    methods_to_evaluate.append(name)
                    seen_method_names.add(name)

        for method_name in methods_to_evaluate:
            if method_name not in self.programs:
                print(f"\nSkipping {method_name} (no program available)")
                continue
            
            program = self.programs[method_name]
            print(f"\n{'='*60}")
            print(f"EVALUATING: {method_name.upper()}")
            print(f"{'='*60}")

            if not self.is_re_mode:
                clear_lm_history(self.main_lm, f"before {method_name} eval")
            
            # Validation evaluation
            val_accuracies = []
            val_detailed = []
            if not self.skip_validation:
                print(f"\nValidation Set ({len(self.val_set) if self.is_re_mode else len(self.taxonomy_examples)} {'episodes' if self.is_re_mode else 'problems'} × {self.eval_runs} runs):")
                if self.is_re_mode:
                    val_accuracies, val_detailed = run_re_episode_set_evaluation(
                        program=program,
                        method_name=method_name,
                        split_name="validation",
                        dataset_type=self.dataset,
                        episodes_data=self.re_dev_data,
                        num_runs=self.eval_runs,
                        model=self.local_model,
                        tokenizer=self.local_tokenizer,
                        batch_size=self.eval_batch_size,
                        yes_token_id=self.yes_token_id,
                        no_token_id=self.no_token_id,
                        log_every=self.log_every,
                        n_chunks=self.re_eval_n_chunks,
                        query_index=self.query_index,
                        output_dir=self.output_dir,
                    )
                else:
                    val_accuracies, val_detailed = run_multi_run_evaluation(
                        program=program,
                        dataset=self.taxonomy_examples,
                        metric=self.metric,
                        num_runs=self.eval_runs,
                        num_threads=self.num_threads
                    )
                
                if val_accuracies:
                    val_stats = compute_per_run_stats(val_accuracies)
                    print(f"  Result: {format_accuracy_with_ci(val_stats)}")
                else:
                    print(f"  Result: No validation results (skipped or empty)")
            else:
                print(f"\nSkipping validation evaluation (--skip_validation)")
            
            # Test evaluation (skip in ablation mode)
            test_accuracies = []
            test_detailed = []
            if not skip_test_evaluation and not self.ablation_mode:
                print(f"\nTest Set ({len(self.test_set)} {'episodes' if self.is_re_mode else 'problems'} × {self.test_runs} runs):")
                if self.is_re_mode:
                    test_accuracies, test_detailed = run_re_episode_set_evaluation(
                        program=program,
                        method_name=method_name,
                        split_name="test",
                        dataset_type=self.dataset,
                        episodes_data=self.re_test_data,
                        num_runs=self.test_runs,
                        model=self.local_model,
                        tokenizer=self.local_tokenizer,
                        batch_size=self.eval_batch_size,
                        yes_token_id=self.yes_token_id,
                        no_token_id=self.no_token_id,
                        log_every=self.log_every,
                        n_chunks=self.re_eval_n_chunks,
                        query_index=self.query_index,
                        output_dir=self.output_dir,
                    )
                else:
                    test_accuracies, test_detailed = run_multi_run_evaluation(
                        program=program,
                        dataset=self.test_set,
                        metric=self.metric,
                        num_runs=self.test_runs,
                        num_threads=self.num_threads
                    )
                
                if test_accuracies:
                    test_stats = compute_per_run_stats(test_accuracies)
                    print(f"  Result: {format_accuracy_with_ci(test_stats)}")
                else:
                    test_stats = compute_per_run_stats([])
                    print(f"  Result: No test results (skipped or empty)")
            else:
                print(f"\nSkipping test evaluation")
                test_stats = compute_per_run_stats([])
            
            # Store results
            val_stats = compute_per_run_stats(val_accuracies) if val_accuracies else None
            test_stats = compute_per_run_stats(test_accuracies) if test_accuracies else None
            
            self.method_results[method_name] = {}
            
            if val_stats:
                self.method_results[method_name]["validation"] = MethodResult(
                    method_name=method_name,
                    per_run_accuracies=val_accuracies,
                    mean_accuracy=val_stats["mean"],
                    ci_lower=val_stats["ci_lower"],
                    ci_upper=val_stats["ci_upper"],
                    std=val_stats["std"],
                    min_acc=val_stats["min"],
                    max_acc=val_stats["max"],
                    num_runs=val_stats["num_runs"],
                    detailed_results=val_detailed
                )
            
            if test_stats and test_stats["num_runs"] > 0:
                self.method_results[method_name]["test"] = MethodResult(
                    method_name=method_name,
                    per_run_accuracies=test_accuracies,
                    mean_accuracy=test_stats["mean"],
                    ci_lower=test_stats["ci_lower"],
                    ci_upper=test_stats["ci_upper"],
                    std=test_stats["std"],
                    min_acc=test_stats["min"],
                    max_acc=test_stats["max"],
                    num_runs=test_stats["num_runs"],
                    detailed_results=test_detailed
                )
            
            phase_stats = {} if self.is_re_mode else convert_dspy_history_to_phase_stats(self.main_lm, self.main_model)
            self._record_phase_stats(f"{method_name}_evaluation", phase_stats)
        
        # V2: Category-level analysis
        if (self.track_category_level and 
            self.trace_assignments and 
            "baseline" in self.method_results and 
            "taxonomy" in self.method_results):
            
            baseline_val = self.method_results["baseline"].get("validation")
            taxonomy_val = self.method_results["taxonomy"].get("validation")
            
            if baseline_val and taxonomy_val:
                selected_names = {c.category_name for c in self.selected_categories}
                
                self.category_level_results = compute_category_level_results(
                    trace_assignments=self.trace_assignments,
                    selected_category_names=selected_names,
                    baseline_detailed=baseline_val.detailed_results,
                    taxonomy_detailed=taxonomy_val.detailed_results,
                    num_eval_runs=self.eval_runs
                )
                
                print_category_level_analysis(self.category_level_results)
    
    def compute_comparisons(self) -> Dict[str, Any]:
        """Compute statistical comparisons between methods."""
        print(f"\n{'='*70}")
        print("STATISTICAL COMPARISON")
        print(f"{'='*70}")
        
        comparisons = {}
        
        if "baseline" not in self.method_results:
            print("No baseline results for comparison")
            return comparisons
        
        baseline_val = self.method_results["baseline"].get("validation")
        baseline_test = self.method_results["baseline"].get("test")
        
        for method_name, results in self.method_results.items():
            if method_name == "baseline":
                continue
            
            print(f"\n{method_name.upper()} vs BASELINE:")
            
            method_val = results.get("validation")
            method_test = results.get("test")
            
            comparisons[method_name] = {}
            
            # Validation comparison
            if baseline_val and method_val:
                val_diff = compute_paired_difference_ci(
                    baseline_val.per_run_accuracies,
                    method_val.per_run_accuracies
                )
                val_mcnemar = mcnemar_test(
                    baseline_val.detailed_results,
                    method_val.detailed_results,
                    self.alpha
                )
                
                print(f"\n  Validation:")
                print(f"    Baseline: {format_accuracy_with_ci(compute_per_run_stats(baseline_val.per_run_accuracies))}")
                print(f"    {method_name}: {format_accuracy_with_ci(compute_per_run_stats(method_val.per_run_accuracies))}")
                print(f"    Δ: {format_diff_with_ci(val_diff)}")
                print(f"    McNemar: p={val_mcnemar['p_value']:.4f} ({'✓ Sig' if val_mcnemar['is_significant'] else '✗ NS'})")
                
                comparisons[method_name]["validation"] = {
                    "paired_difference": val_diff,
                    "mcnemar": val_mcnemar
                }
            else:
                print(f"\n  Validation: skipped")
            
            # Test comparison
            if baseline_test and method_test:
                test_diff = compute_paired_difference_ci(
                    baseline_test.per_run_accuracies,
                    method_test.per_run_accuracies
                )
                test_mcnemar = mcnemar_test(
                    baseline_test.detailed_results,
                    method_test.detailed_results,
                    self.alpha
                )
                
                print(f"\n  Test:")
                print(f"    Baseline: {format_accuracy_with_ci(compute_per_run_stats(baseline_test.per_run_accuracies))}")
                print(f"    {method_name}: {format_accuracy_with_ci(compute_per_run_stats(method_test.per_run_accuracies))}")
                print(f"    Δ: {format_diff_with_ci(test_diff)}")
                print(f"    McNemar: p={test_mcnemar['p_value']:.4f} ({'✓ Sig' if test_mcnemar['is_significant'] else '✗ NS'})")
                
                comparisons[method_name]["test"] = {
                    "paired_difference": test_diff,
                    "mcnemar": test_mcnemar
                }
            else:
                print(f"\n  Test: skipped")
        
        return comparisons
    
    def save_results(self, comparisons: Dict[str, Any]):
        """Save all results to files."""
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")
        
        results_data = {}
        for method_name, split_results in self.method_results.items():
            results_data[method_name] = {}
            for split_name, result in split_results.items():
                results_data[method_name][split_name] = {
                    "mean_accuracy": result.mean_accuracy,
                    "ci_lower": result.ci_lower,
                    "ci_upper": result.ci_upper,
                    "std": result.std,
                    "min": result.min_acc,
                    "max": result.max_acc,
                    "num_runs": result.num_runs,
                    "per_run_accuracies": result.per_run_accuracies
                }
        
        # V2: Category-level results
        category_level_data = []
        if self.category_level_results:
            for r in self.category_level_results:
                category_level_data.append({
                    "category_name": r.category_name,
                    "num_problems": r.num_problems,
                    "problem_indices": r.problem_indices,
                    "baseline_failure_rate": r.baseline_failure_rate,
                    "taxonomy_failure_rate": r.taxonomy_failure_rate,
                    "delta_failure_rate": r.delta_failure_rate,
                    "was_selected": r.was_selected,
                    "baseline_failures": r.baseline_failures,
                    "taxonomy_failures": r.taxonomy_failures,
                    "total_evals": r.total_evals
                })
        
        final_results = {
            "experiment_name": self.expr_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "dataset": self.dataset,
                "task_mode": self.task_mode,
                "valid_size": self.valid_size,
                "methods": self.methods,
                "main_model": self.main_model,
                "taxonomy_runs": self.taxonomy_runs,
                "coverage_threshold": self.coverage_threshold,
                "max_guidances": self.max_guidances,
                "min_problems": self.min_problems,
                "prompt_style": self.prompt_style,
                "eval_runs": self.eval_runs,
                "test_runs": self.test_runs,
                "optimizer_auto": self.optimizer_auto,
                "seed": self.seed,
                "skip_validation": self.skip_validation,
                "ablation_mode": self.ablation_mode,
                "load_taxonomy": str(self.load_taxonomy_path) if self.load_taxonomy_path else None
            },
            "results": results_data,
            "comparisons": comparisons,
            "category_level_analysis": category_level_data,
            "token_stats": self.token_stats
        }
        
        results_file = self.output_dir / "final_results.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Output directory: {self.output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        print(f"\n{'='*80}")
        print("UNIFIED PROMPT OPTIMIZATION COMPARISON")
        print(f"{'='*80}")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"Dataset: {self.dataset}")
        
        if self.ablation_mode:
            print(f"Mode: ABLATION (skip test evaluation)")
        if self.raw_sampling:
            print(f"Mode: RAW SAMPLING (skip taxonomy, sample {self.num_raw_samples} raw traces)")
        if self.direct_categories:
            print(f"Mode: DIRECT CATEGORIES (skip guidance generation)")
        if self.is_re_mode:
            print(f"Task mode: {self.task_mode}")
        if self.load_taxonomy_path:
            print(f"Loading taxonomy from: {self.load_taxonomy_path}")
        
        start_time = time.time()
        
        # Setup
        self.setup()

        if self.is_re_mode and self.raw_sampling:
            raise ValueError("raw_sampling is not supported in relation_extraction_non_reasoning mode")
        if self.is_re_mode and self.direct_categories:
            raise ValueError("direct_categories is not supported in relation_extraction_non_reasoning mode")
        
        # Run taxonomy method if selected
        if "taxonomy" in self.methods:
            if self.raw_sampling:
                # Ablation 1: Raw failure sampling (skip taxonomy)
                failures = self.run_baseline_for_taxonomy()
                if failures:
                    self.generate_guidance_from_raw_samples(failures)
            elif self.load_taxonomy_path and self.load_taxonomy_path.exists():
                # V2: Load existing taxonomy
                print(f"\n{'='*70}")
                print("TAXONOMY: LOADING FROM FILE")
                print(f"{'='*70}")
                
                (self.issue_categories,
                 self.trace_assignments,
                 self.failure_records,
                 loaded_metadata) = load_taxonomy_from_file(self.load_taxonomy_path)
                
                print(f"Loaded metadata: {loaded_metadata}")
                
                # Select categories with current T, G values
                self.select_categories()
                
                if self.direct_categories:
                    # Ablation 2: Direct category insertion
                    self.generate_direct_category_prompt()
                else:
                    self.generate_guidance()
            else:
                # Build taxonomy from scratch
                failures = self.run_baseline_for_taxonomy()
                if failures:
                    self.build_taxonomy(failures)
                    self.select_categories()
                    
                    if self.direct_categories:
                        # Ablation 2: Direct category insertion
                        self.generate_direct_category_prompt()
                    else:
                        self.generate_guidance()
        
        # Run GEPA if selected (skip in ablation mode)
        if "gepa" in self.methods and not self.ablation_mode:
            self.run_gepa_optimization()
        
        # Run MIPROv2 if selected (skip in ablation mode)
        if "mipro" in self.methods and not self.ablation_mode:
            self.run_mipro_optimization()
        
        # Evaluate all methods
        self.evaluate_all_methods()
        
        # Compute comparisons
        comparisons = self.compute_comparisons()
        
        # Save results
        self.save_results(comparisons)
        
        # Print final token summary
        print_final_token_summary(self.token_stats)
        
        elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*80}")
        print(f"COMPLETE (elapsed: {elapsed:.1f} minutes)")
        print(f"{'='*80}")
        
        return {
            "results": self.method_results,
            "comparisons": comparisons,
            "category_level_results": self.category_level_results
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Prompt Optimization Comparison using DSPy (V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods
  python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy,gepa,mipro

  # Run only taxonomy vs baseline
  python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy

  # Ablation mode: taxonomy only, skip test eval
  python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy \\
      --ablation_mode --taxonomy_runs 5

  # Factored execution: Load existing taxonomy and vary T/G
  python -u -m etgpo_full --dataset aimo --methods baseline,taxonomy \\
      --load_taxonomy path/to/taxonomy.json --coverage_threshold 0.5 --max_guidances 5 --ablation_mode
        """
    )
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='aimo',
                        help='Dataset name (default: aimo)')
    parser.add_argument('--task_mode', type=str, default='',
                        help=f'Optional task mode. Use {RE_TASK_MODE} for TACRED-style binary RE.')
    parser.add_argument('--valid_size', type=int, default=90,
                        help='Number of validation examples (default: 90)')
    
    # Methods
    parser.add_argument('--methods', type=str, default='baseline,taxonomy',
                        help='Comma-separated list of methods: baseline,taxonomy,gepa,mipro')
    
    # Models
    parser.add_argument('--main_model', type=str, default=DEFAULT_MAIN_MODEL,
                        help=f'Main inference model (default: {DEFAULT_MAIN_MODEL})')
    parser.add_argument('--device_map', type=str, default=None,
                        help='Device map for local HF models in RE mode (e.g., cuda:0, cuda:1, cpu, auto)')
    parser.add_argument('--main_temperature', type=float, default=DEFAULT_MAIN_TEMPERATURE,
                        help=f'Main model temperature (default: {DEFAULT_MAIN_TEMPERATURE})')
    parser.add_argument('--reflection_model', type=str, default=DEFAULT_REFLECTION_MODEL,
                        help=f'Reflection model for GEPA (default: {DEFAULT_REFLECTION_MODEL})')
    parser.add_argument('--taxonomy_model', type=str, default=DEFAULT_TAXONOMY_MODEL,
                        help=f'Model for taxonomy analysis (default: {DEFAULT_TAXONOMY_MODEL})')
    parser.add_argument('--max_new_tokens', type=int, default=None,
                        help='Max new tokens for taxonomy/guidance JSON generation. Defaults to 10000 in RE mode and 32768 otherwise.')
    
    # Taxonomy settings
    parser.add_argument('--taxonomy_runs', type=int, default=5,
                        help='Runs for taxonomy failure collection (default: 5)')
    parser.add_argument('--taxonomy_batch_size', type=int, default=6,
                        help='Failures per taxonomy LLM call (default: 6)')
    parser.add_argument('--include_solution', action='store_true',
                        help='Include ground truth solution in taxonomy')
    parser.add_argument('--coverage_threshold', type=float, default=0.7,
                        help='Coverage threshold for category selection (default: 0.7)')
    parser.add_argument('--max_guidances', type=int, default=7,
                        help='Maximum guidance categories (default: 7)')
    parser.add_argument('--min_problems', type=int, default=2,
                        help='Minimum problems per category (default: 2)')
    parser.add_argument('--prompt_style', type=str, default='detailed',
                        choices=['short', 'detailed'],
                        help='Generated prompt style (default: detailed)')
    parser.add_argument('--include_traces', action='store_true',
                        help='Include sample traces in guidance generation')
    parser.add_argument('--num_sample_traces', type=int, default=2,
                        help='Sample traces per category (default: 2)')
    parser.add_argument('--num_guidance_prompts', type=int, default=1,
                        help='Number of separate guidance-generation calls/prompts to produce (default: 1)')
    
    # Evaluation settings
    parser.add_argument('--eval_runs', type=int, default=30,
                        help='Runs for validation evaluation (default: 30)')
    parser.add_argument('--test_runs', type=int, default=30,
                        help='Runs for test evaluation (default: 30)')
    parser.add_argument('--num_threads', type=int, default=64,
                        help='Parallel threads (default: 64)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05)')
    
    # Optimizer settings
    parser.add_argument('--optimizer_auto', type=str, default='heavy',
                        help='GEPA/MIPROv2 auto setting (default: heavy)')
    parser.add_argument('--train_frac', type=float, default=0.5,
                        help='Fraction of val set for training GEPA/MIPROv2 (default: 0.5)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='unified_optimization_results/',
                        help='Output directory (default: unified_optimization_results/)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip validation evaluation (only run test evaluation)')

    # Relation extraction mode
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory for relation extraction mode')
    parser.add_argument('--train_samples_file', type=str, default='fs_tacred_train_samples.pkl',
                        help='Train-sample pickle for relation extraction mode')
    parser.add_argument('--dev_split', type=str, default='dev',
                        help='Validation split name for relation extraction mode')
    parser.add_argument('--test_split', type=str, default='test',
                        help='Test split name for relation extraction mode')
    parser.add_argument('--query_index', type=int, default=0,
                        help='Query index for relation extraction episode evaluation')
    parser.add_argument('--re_inference_mode', type=str, default=INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
                        help='Inference prompt mode for relation extraction')
    parser.add_argument('--re_feedback_prompt_key', type=str, default='mistakes_v1',
                        help='Feedback prompt key for relation extraction')
    parser.add_argument('--inference_batch_size', type=int, default=8,
                        help='Binary inference batch size for relation extraction mode')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='Evaluation batch size for relation extraction mode')
    parser.add_argument('--feedback_batch_size', type=int, default=4,
                        help='Feedback generation batch size for relation extraction mode')
    parser.add_argument('--feedback_max_new_tokens', type=int, default=10000,
                        help='Feedback generation max new tokens for relation extraction mode')
    parser.add_argument('--log_every', type=int, default=20,
                        help='Logging frequency for local binary inference')
    parser.add_argument('--re_eval_n_chunks', type=int, default=3,
                        help='Number of chunks for RE F1 aggregation (default: 3)')
    
    # V2: Factored execution
    parser.add_argument('--load_taxonomy', type=str, default=None,
                        help='Load taxonomy from file (skip failure collection and taxonomy building)')
    parser.add_argument('--save_taxonomy', type=str, default=None,
                        help='Save taxonomy to file for later reuse')
    parser.add_argument('--ablation_mode', action='store_true',
                        help='Ablation mode: skip GEPA/MIPROv2, skip test evaluation')
    parser.add_argument('--no_category_tracking', action='store_true',
                        help='Disable category-level tracking')
    
    # V2: Method ablations
    parser.add_argument('--raw_sampling', action='store_true',
                        help='Ablation: Skip taxonomy, sample raw failure traces directly')
    parser.add_argument('--num_raw_samples', type=int, default=10,
                        help='Number of raw failure traces to sample (default: 10)')
    parser.add_argument('--direct_categories', action='store_true',
                        help='Ablation: Skip guidance generation, insert category descriptions directly')
    
    args = parser.parse_args()
    
    # Parse methods
    methods = [m.strip() for m in args.methods.split(',')]
    
    optimizer = UnifiedPromptOptimizer(
        dataset=args.dataset,
        task_mode=args.task_mode,
        valid_size=args.valid_size,
        methods=methods,
        main_model=args.main_model,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        main_temperature=args.main_temperature,
        reflection_model=args.reflection_model,
        taxonomy_model=args.taxonomy_model,
        taxonomy_runs=args.taxonomy_runs,
        taxonomy_batch_size=args.taxonomy_batch_size,
        include_solution=args.include_solution,
        coverage_threshold=args.coverage_threshold,
        max_guidances=args.max_guidances,
        min_problems=args.min_problems,
        prompt_style=args.prompt_style,
        include_traces=args.include_traces,
        num_sample_traces=args.num_sample_traces,
        num_guidance_prompts=args.num_guidance_prompts,
        eval_runs=args.eval_runs,
        test_runs=args.test_runs,
        num_threads=args.num_threads,
        alpha=args.alpha,
        optimizer_auto=args.optimizer_auto,
        train_frac=args.train_frac,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        verbose=args.verbose,
        skip_validation=args.skip_validation,
        load_taxonomy=args.load_taxonomy,
        save_taxonomy=args.save_taxonomy,
        ablation_mode=args.ablation_mode,
        track_category_level=not args.no_category_tracking,
        raw_sampling=args.raw_sampling,
        num_raw_samples=args.num_raw_samples,
        direct_categories=args.direct_categories,
        data_dir=args.data_dir,
        train_samples_file=args.train_samples_file,
        dev_split=args.dev_split,
        test_split=args.test_split,
        query_index=args.query_index,
        re_inference_mode=args.re_inference_mode,
        re_feedback_prompt_key=args.re_feedback_prompt_key,
        inference_batch_size=args.inference_batch_size,
        eval_batch_size=args.eval_batch_size,
        feedback_batch_size=args.feedback_batch_size,
        feedback_max_new_tokens=args.feedback_max_new_tokens,
        log_every=args.log_every,
        re_eval_n_chunks=args.re_eval_n_chunks
    )
    
    optimizer.run()


if __name__ == "__main__":
    main()
