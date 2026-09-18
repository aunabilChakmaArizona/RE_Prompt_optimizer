"""Test input-token objectives and beam score caching without a GPU engine."""

from __future__ import annotations

import math
import json
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from agents.agent_vllm_models import _validate_vllm_version
from benchmark_vllm_objective_scoring import load_benchmark_examples
from prompt_optimization.cli_common import build_context
from prompt_optimization.qa_task import resolve_mode
from prompt_optimization.second_stage import _beam_search_replacements, _score_objective_candidates
from prompt_optimization.sequence_gradients import score_combined_objectives
from prompt_optimization.vllm_scoring import (
    encode_scoring_sequences,
    score_combined_objectives_vllm,
    score_token_sequences,
    validate_vllm_scoring_version,
)


class CharacterTokenizer:
    """Expose the same character IDs and offsets for both objective backends."""

    padding_side = "left"

    def apply_chat_template(self, messages, **_kwargs):
        """Join messages without adding or removing answer text."""
        return "\n".join(message["content"] for message in messages)

    def __call__(self, texts, **kwargs):
        """Return unpadded lists or left-padded HF tensors with original offsets."""
        texts = [texts] if isinstance(texts, str) else list(texts)
        width = max(len(text) for text in texts)
        ids, masks, offsets = [], [], []
        for text in texts:
            padding = width - len(text) if kwargs.get("padding") else 0
            ids.append([0] * padding + [ord(character) for character in text])
            masks.append([0] * padding + [1] * len(text))
            offsets.append([(0, 0)] * padding + [(index, index + 1) for index in range(len(text))])
        result = {"input_ids": ids, "attention_mask": masks, "offset_mapping": offsets}
        if kwargs.get("return_tensors") == "pt":
            result = {key: torch.tensor(value) for key, value in result.items()}
        return result


class FakeVLLM:
    """Return actual-token logprobs from a deterministic causal distribution."""

    def __init__(self):
        """Prepare request capture and cached distributions for each context."""
        self.calls = []
        self.distributions = {}

    def generate(self, prompts, *, sampling_params, use_tqdm):
        """Score each supplied token from its preceding context, not itself."""
        self.calls.append((prompts, sampling_params))
        outputs = []
        for prompt in prompts:
            tokens = list(prompt["prompt_token_ids"])
            context_sum = tokens[0]
            logprobs = [None]
            for token in tokens[1:]:
                center = context_sum % 128
                if center not in self.distributions:
                    logits = [-abs(index - center) / 20 + index * 0.01 for index in range(128)]
                    normalizer = math.log(math.fsum(math.exp(value) for value in logits))
                    self.distributions[center] = [value - normalizer for value in logits]
                value = self.distributions[center][token]
                logprobs.append({token: SimpleNamespace(logprob=value)})
                context_sum += token
            outputs.append(SimpleNamespace(
                prompt_token_ids=tokens,
                prompt_logprobs=logprobs,
                outputs=[SimpleNamespace(text="IGNORED", logprobs=[-999])],
            ))
        return outputs


class FakeHF:
    """Produce the same causal distribution as FakeVLLM with left padding."""

    def get_input_embeddings(self):
        """Expose a CPU model device for the HF scoring helper."""
        return SimpleNamespace(weight=torch.zeros(1))

    def __call__(self, *, input_ids, use_cache, attention_mask=None):
        """Return logits whose centers depend on all preceding real token IDs."""
        centers = (input_ids.cumsum(dim=1) % 128).unsqueeze(-1)
        vocabulary = torch.arange(128, dtype=torch.float32)
        logits = -torch.abs(vocabulary - centers) / 20 + vocabulary * 0.01
        return SimpleNamespace(logits=logits)


class VLLMObjectiveTests(unittest.TestCase):
    """Check exact targets, averaging, batching, backend routing, and caching."""

    def setUp(self):
        """Install a lightweight SamplingParams stub for GPU-independent tests."""
        fake_module = ModuleType("vllm")
        fake_module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
        self.modules = patch.dict(sys.modules, {"vllm": fake_module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.version = patch("importlib.metadata.version", return_value="0.11.0")
        self.version.start()
        self.addCleanup(self.version.stop)
        self.mode = resolve_mode("reasoning", "math500")
        self.records = [
            {"id": "one", "question": "Compute 2/2.", "answer": "1"},
            {"id": "two", "question": "Compute 1/2.", "answer": r"\frac{1}{2}"},
        ]
        self.traces = ["Reasoning one. <answer>7</answer>", "Reasoning two. <answer>9</answer>"]

    def test_version_guard_preserves_old_environment_but_requires_v2_scoring(self):
        """Accept both loaders while restricting the new scorer to its tested release."""
        for version in ("0.8.5.post1", "0.11.0"):
            with patch("importlib.metadata.version", return_value=version):
                self.assertEqual(_validate_vllm_version(), version)
        with patch("importlib.metadata.version", return_value="0.8.5.post1"):
            with self.assertRaisesRegex(RuntimeError, "0.11.0"):
                validate_vllm_scoring_version()

    def test_gold_target_replaces_prediction_and_preserves_latex(self):
        """Select only the exact gold answer, not reasoning or answer tags."""
        tokens, positions = encode_scoring_sequences(
            ["Solve carefully."] * 2, self.records, mode=self.mode,
            tokenizer=CharacterTokenizer(), reasoning_traces=self.traces,
        )
        self.assertEqual(["".join(chr(row[index]) for index in selected) for row, selected in zip(tokens, positions)], ["1", r"\frac{1}{2}"])
        text = "".join(chr(token) for token in tokens[1])
        self.assertIn("Reasoning two.", text)
        self.assertIn(r"<answer>\frac{1}{2}</answer>", text)
        self.assertNotIn("<answer>9</answer>", text)

    def test_openbookqa_and_hotpotqa_reuse_their_existing_renderers(self):
        """Keep label casing for direct QA and multi-token open-QA targets."""
        for task, mode_name, record, trace, gold in (
            ("openbookqa", "non_reasoning", {"id": "qa", "question": "Which?", "choices": [{"label": "A", "text": "Yes"}, {"label": "B", "text": "No"}], "answer": "a"}, None, "A"),
            ("hotpotqa", "reasoning", {"id": "hotpot", "question": "Where?", "context": [], "answer": "New York"}, "Reasoning. <answer>Paris</answer>", "New York"),
        ):
            with self.subTest(task=task):
                tokens, positions = encode_scoring_sequences(
                    ["Answer carefully."], [record], mode=resolve_mode(mode_name, task),
                    tokenizer=CharacterTokenizer(), reasoning_traces=[trace],
                )
                self.assertEqual("".join(chr(tokens[0][index]) for index in positions[0]), gold)

    def test_invalid_runtime_settings_fail_before_loading_data_or_models(self):
        """Require dual mode for vLLM scoring and reject invalid submission sizes."""
        for args, message in (
            (SimpleNamespace(backend="transformers", objective_scoring_backend="vllm"), "requires --backend dual"),
            (SimpleNamespace(objective_scoring_batch_size=0), "must be positive"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                build_context(args, "gradpo_gen")

    def test_benchmark_loads_math_as_math_without_regenerating_answers(self):
        """Restore cached traces read-only with the proper symbolic record validator."""
        payload = {
            "metadata": {"model_id": "target", "task_name": "math500", "mode_name": "reasoning"},
            "pool_results": [dict(record_id=record["id"], raw_response=trace, gold_answer=record["answer"]) for record, trace in zip(self.records, self.traces)],
        }
        args = SimpleNamespace(cache_file="unused.json", train_path="unused.jsonl", model="target", qa_task="math500", qa_mode="reasoning", sample_size=2)
        with patch("pathlib.Path.read_text", return_value=json.dumps(payload)), patch("benchmark_vllm_objective_scoring.load_qa_records", return_value=self.records) as loader:
            records, traces = load_benchmark_examples(args)
        self.assertEqual(set(record["id"] for record in records), {"one", "two"})
        self.assertEqual(set(traces), set(self.traces))
        loader.assert_called_once_with("unused.jsonl", expected_task="math500")

    def test_task_loss_means_per_problem_and_fluency_uses_raw_instruction(self):
        """Keep long answers from dominating and score fluency without a template."""
        model = FakeVLLM()
        prompts = ["Solve.", "Check carefully.", "X"]
        results = score_combined_objectives_vllm(
            prompts, self.records, mode=self.mode, model=model,
            tokenizer=CharacterTokenizer(), batch_size=4, fluency_lambda=0.5,
            reasoning_traces=self.traces,
        )
        self.assertEqual([len(call[0]) for call in model.calls], [4, 2, 2])
        for _, params in model.calls:
            self.assertEqual(params.prompt_logprobs, 0)
            self.assertEqual(params.max_tokens, 1)
        self.assertEqual(model.calls[-1][0][0]["prompt_token_ids"], [ord(character) for character in prompts[0]])
        tokens, positions = encode_scoring_sequences(
            [prompts[0]] * 2, self.records, mode=self.mode,
            tokenizer=CharacterTokenizer(), reasoning_traces=self.traces,
        )
        per_problem = score_token_sequences(model, tokens, positions)
        self.assertAlmostEqual(results[0]["task_loss"], sum(per_problem) / 2)
        self.assertEqual(results[2]["instruction_nll"], 0.0)
        for result in results:
            self.assertAlmostEqual(result["combined_score"], result["task_loss"] + 0.5 * result["instruction_nll"])

    @unittest.skipIf(torch is None, "HF parity requires PyTorch.")
    def test_hf_and_vllm_objectives_agree_for_identical_causal_logits(self):
        """Verify token shifts, left padding, per-example weighting, and rankings."""
        prompts = ["Solve.", "Solve and check.", "X"]
        kwargs = dict(mode=self.mode, tokenizer=CharacterTokenizer(), batch_size=4,
                      fluency_lambda=0.5, reasoning_traces=self.traces)
        hf = score_combined_objectives(prompts, self.records, model=FakeHF(), **kwargs)
        vllm = score_combined_objectives_vllm(prompts, self.records, model=FakeVLLM(), **kwargs)
        for first, second in zip(hf, vllm):
            for field in ("task_loss", "instruction_nll", "combined_score"):
                self.assertAlmostEqual(first[field], second[field], places=5)
        self.assertEqual(sorted(range(3), key=lambda index: hf[index]["combined_score"]), sorted(range(3), key=lambda index: vllm[index]["combined_score"]))

    def test_missing_nonfinite_or_retokenized_logprobs_fail_explicitly(self):
        """Do not silently replace missing input probabilities with output scores."""
        for output in (
            SimpleNamespace(prompt_token_ids=[65, 66], prompt_logprobs=None),
            SimpleNamespace(prompt_token_ids=[65, 66], prompt_logprobs=[None, {}]),
            SimpleNamespace(prompt_token_ids=[65, 66], prompt_logprobs=[None, {66: SimpleNamespace(logprob=float("nan"))}]),
            SimpleNamespace(prompt_token_ids=[65, 67], prompt_logprobs=[None, {}]),
        ):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                score_token_sequences(SimpleNamespace(generate=lambda *_args, **_kwargs: [output]), [[65, 66]], [[1]])

    def test_dispatch_uses_target_vllm_without_hf_forward(self):
        """Route the shared GreaTer/GradPO objective call to the target engine."""
        model = object()
        context = SimpleNamespace(
            args=SimpleNamespace(objective_scoring_backend="vllm", objective_scoring_batch_size=128),
            model_pool=SimpleNamespace(ensure_vllm=MagicMock(return_value=(model, object()))),
            mode=self.mode, logger=SimpleNamespace(event=MagicMock()),
            optimizer_name="gradpo_gen", started_at=time.monotonic(),
        )
        with patch("prompt_optimization.vllm_scoring.score_combined_objectives_vllm", return_value=[{"combined_score": 1.0}]) as scorer:
            result = _score_objective_candidates(context, ["Solve."], self.records, model=object(), tokenizer=CharacterTokenizer(), batch_size=1, fluency_lambda=0.5, reasoning_traces=self.traces)
        self.assertEqual(result, [{"combined_score": 1.0}])
        self.assertIs(scorer.call_args.kwargs["model"], model)
        self.assertEqual(scorer.call_args.kwargs["batch_size"], 128)
        context.model_pool.ensure_vllm.assert_called_once_with("target")

    def test_beam_does_not_rescore_unchanged_survivors(self):
        """Cache exact objectives only within one fixed beam-search context."""
        context = SimpleNamespace(
            initial_prompt="Solve carefully and check.", mode=self.mode,
            optimizer_name="gradpo_gen", started_at=time.monotonic(),
            args=SimpleNamespace(objective_scoring_backend="transformers"),
            logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        )
        regions = [
            dict(region_rank=1, region_text="carefully", start_char=6, end_char=15),
            dict(region_rank=2, region_text="check", start_char=20, end_char=25),
        ]
        candidates = [dict(region_rank=1, candidates=["carefully", "exactly"]),
                      dict(region_rank=2, candidates=["check", "verify"])]
        def objective(_context, prompts, _records, **_kwargs):
            """Assign stable deterministic objectives for all new prompt strings."""
            return [dict(task_loss=1, instruction_nll=1, combined_score=1) for _ in prompts]
        with patch("prompt_optimization.second_stage._score_objective_candidates", side_effect=objective) as scorer:
            _, trace = _beam_search_replacements(
                context, regions, candidates, self.records, model=object(), tokenizer=object(),
                beam_width=2, selection_batch_size=1, fluency_lambda=0.5,
                replacement_mode="direct", synthesis_max_new_tokens=100,
                synthesis_batch_size=1, reasoning_traces=self.traces,
            )
        self.assertEqual([len(call.args[1]) for call in scorer.call_args_list], [2, 2])
        self.assertEqual(trace[1]["objective_cache_hits"], 2)
        self.assertEqual(trace[1]["expansion_count"], 4)


if __name__ == "__main__":
    unittest.main()
