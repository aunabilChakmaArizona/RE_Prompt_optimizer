from __future__ import annotations

from typing import Dict, List, Sequence


def get_sentence_with_tags(meta_data: Dict) -> str:
    token_with_tags = list(meta_data["token"])
    token_with_tags[meta_data["subj_start"]] = (
        "<subject>" + token_with_tags[meta_data["subj_start"]]
    )
    token_with_tags[meta_data["subj_end"]] = (
        token_with_tags[meta_data["subj_end"]] + "</subject>"
    )
    token_with_tags[meta_data["obj_start"]] = (
        "<object>" + token_with_tags[meta_data["obj_start"]]
    )
    token_with_tags[meta_data["obj_end"]] = (
        token_with_tags[meta_data["obj_end"]] + "</object>"
    )
    return " ".join(token_with_tags)


def build_support_block(support_sentences: Sequence[str]) -> str:
    if not support_sentences:
        return ""
    if len(support_sentences) == 1:
        return f"Support Sentence: {support_sentences[0]}"
    lines = []
    for idx, sent in enumerate(support_sentences, start=1):
        lines.append(f"Support Sentence {idx}: {sent}")
    return "\n".join(lines)


def resolve_way_shots(way_ids: Sequence[int], shots: Dict) -> List[Dict]:
    resolved = []
    if not way_ids:
        return resolved
    resolved.append(shots["shots"][way_ids[0]])

    # may come from LLM generation / UMBC index
    # for shot_id in way_ids[1:]:
    #     resolved.append(shots["umbc_shots"][shot_id])
    
    return resolved
