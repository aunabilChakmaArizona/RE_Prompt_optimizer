"""Run chat-formatted prompts with vLLM continuous batching."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from agents.agent_token_usage import TokenUsage, create_token_usage


_SAMPLE_LOGGED_LABELS: set[str] = set()


def _log_first_sample(label: str, prompt: str, output: str) -> None:
    """Print one representative prompt and response for a named run."""
    if label in _SAMPLE_LOGGED_LABELS:
        return
    _SAMPLE_LOGGED_LABELS.add(label)
    print(f"\n[{label}] sample first prompt:\n{prompt}", flush=True)
    print(f"\n[{label}] sample first output:\n{output}", flush=True)


def _supports_qwen_thinking(model_id: str, tokenizer: Any) -> bool:
    """Detect whether the model chat template uses Qwen's thinking switch."""
    if "qwen3" in model_id.casefold():
        return True
    chat_template = getattr(tokenizer, "chat_template", "")
    return "enable_thinking" in str(chat_template)


def _as_token_id_list(tokenized_prompt: Any) -> list[int]:
    """Convert tokenizer list, tensor, or BatchEncoding output to token IDs."""
    if isinstance(tokenized_prompt, Mapping):
        tokenized_prompt = tokenized_prompt["input_ids"]
    if hasattr(tokenized_prompt, "tolist"):
        tokenized_prompt = tokenized_prompt.tolist()
    if (
        len(tokenized_prompt) == 1
        and isinstance(tokenized_prompt[0], (list, tuple))
    ):
        tokenized_prompt = tokenized_prompt[0]
    return [int(token_id) for token_id in tokenized_prompt]


def format_vllm_prompts(
    prompts: Sequence[str],
    *,
    model_id: str,
    tokenizer: Any,
    system_message: str | None = None,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
) -> list[dict[str, list[int]]]:
    """Apply the model template once and return pre-tokenized vLLM prompts."""
    formatted_prompts = []
    supports_qwen_thinking = _supports_qwen_thinking(model_id, tokenizer)

    for prompt in prompts:
        if use_chat_template:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            template_kwargs = {}
            if supports_qwen_thinking:
                template_kwargs["enable_thinking"] = enable_thinking
            tokenized_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **template_kwargs,
            )
            prompt_token_ids = _as_token_id_list(tokenized_prompt)
        else:
            prompt_token_ids = _as_token_id_list(
                tokenizer.encode(prompt, add_special_tokens=True)
            )

        formatted_prompts.append({"prompt_token_ids": prompt_token_ids})

    return formatted_prompts


def run_prompts_vllm(
    prompts: Sequence[str],
    *,
    model_id: str,
    model: Any,
    tokenizer: Any,
    system_message: str | None = None,
    max_new_tokens: int = 10000,
    use_chat_template: bool = True,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    do_log: bool = False,
    log_label: str | None = None,
    return_token_usage: bool = False,
) -> list[str] | tuple[list[str], list[TokenUsage]]:
    """Generate all prompts with vLLM-managed continuous batching."""
    if not prompts:
        return ([], []) if return_token_usage else []
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    try:
        from vllm import SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "vLLM is unavailable. Install the pinned requirements_vllm.txt dependencies."
        ) from error

    formatted_prompts = format_vllm_prompts(
        prompts,
        model_id=model_id,
        tokenizer=tokenizer,
        system_message=system_message,
        use_chat_template=use_chat_template,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )
    decoding_parameters = model_default_sampling_parameters(model_id)
    sampling_params = SamplingParams(
        **decoding_parameters,
        max_tokens=max_new_tokens,
        skip_special_tokens=True,
    )

    if do_log:
        print(
            f"[agent_vllm_prompting] submitting {len(formatted_prompts)} prompts "
            f"with continuous batching and sampling={decoding_parameters}"
        )
    request_outputs = model.generate(
        formatted_prompts,
        sampling_params=sampling_params,
        use_tqdm=do_log,
    )
    if len(request_outputs) != len(prompts):
        raise RuntimeError(
            f"vLLM returned {len(request_outputs)} responses for {len(prompts)} prompts."
        )

    outputs = []
    token_usages = []
    for formatted_prompt, request_output in zip(formatted_prompts, request_outputs):
        if not request_output.outputs:
            raise RuntimeError("vLLM returned a request without a generated output.")
        completion_output = request_output.outputs[0]
        outputs.append(completion_output.text)
        if return_token_usage:
            token_usages.append(
                create_token_usage(
                    len(formatted_prompt["prompt_token_ids"]),
                    len(completion_output.token_ids),
                )
            )

    if log_label and outputs:
        _log_first_sample(log_label, prompts[0], outputs[0])
    if return_token_usage:
        return outputs, token_usages
    return outputs
