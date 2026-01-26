from typing import List

class FeedbackSample:
    def __init__(
        self,
        id_1shot: int,
        id_query: int,
        inference: str,
        label: str,
    ):
        self.id_1shot = id_1shot
        self.id_query = id_query
        self.inference = inference
        self.label = label
        self.relation = ""
        self.support_sentence = ""
        self.query_sentence = ""
        self.raw_feedback_text = ""
        self.feedback_text = ""

    def __repr__(self):
        return (
            f"FeedbackSample("
            f"id_1shot={self.id_1shot}, "
            f"id_query={self.id_query})"
        )

class FeedbackSamples:
    def __init__(self):
        # All K samples
        self.all_samples: List[FeedbackSample] = []

        # Selected subset (e.g., 3-shot feedback)
        self.selected_samples: List[FeedbackSample] = []
        self.raw_feedback_texts: List[str] = []
        self.feedback_texts: List[str] = []

    def add_to_all_samples(self, feedback_sample: FeedbackSample):
        self.all_samples.append(feedback_sample)

    def add_to_selected_samples(self, feedback_sample: FeedbackSample):
        self.selected_samples.append(feedback_sample)

    def __repr__(self):
        return (
            f"FeedbackSamples("
            f"K={len(self.all_samples)}, "
            f"selected={len(self.selected_samples)}, "
        )
