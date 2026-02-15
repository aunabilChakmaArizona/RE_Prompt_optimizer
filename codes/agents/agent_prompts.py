FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1 = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing reasoning-based feedback explaining why a relation extraction system arrived at a particular yes/no decision, for both correct and incorrect predictions.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given an input instance for relation inference below that contains:
- A relation and its description
- A support instance (an example where the relation holds)
- A query sentence
- A ground-truth label (yes/no)
- A yes/no inference made by another LLM on this instance

The ground-truth label indicates whether the LLM inference was correct or incorrect and is provided only as contextual information. The feedback model’s task is to explain the most likely reasoning process that led to the model’s answer, not to re-evaluate, judge, or correct the prediction. In particular:
- If the prediction matches the label, explain what cues or evidence likely led to that choice.
- If it does not match, explain what misunderstanding, missing evidence, or heuristic likely caused the prediction.

Instance:
```
Relation: #RELATION#
Relation Description: #RELATION_DESCRIPTION#
Support Instance: #SUPPORT_INSTANCE#

Query: #QUERY#

Label: #LABEL#
LLM Inference: #INFERENCE#
```

You may perform reasoning internally, but provide only your final remarks or official feedback within the <f> and </f> tags.
'''

FEEDBACK_INFERENCE_PROMPT_CORRECT_V1 = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing reasoning-based feedback explaining why a relation extraction system arrived at this correct prediction.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given an input instance for relation inference below that contains:
- A relation and its description
- A support instance (an example where the relation holds)
- A query sentence
- A ground-truth label (yes/no)
- A yes/no inference made by another LLM on this instance

The ground-truth label is provided only as contextual information. The feedback model’s task is to explain the most likely reasoning process that led to the model’s correct yes/no decision, highlighting the cues or evidence that support it.

Instance:
```
Relation: #RELATION#
Relation Description: #RELATION_DESCRIPTION#
Support Instance: #SUPPORT_INSTANCE#

Query: #QUERY#

Label: #LABEL#
LLM Inference: #INFERENCE#
```

You may perform reasoning internally, but provide only your final remarks or official feedback within the <f> and </f> tags.
'''

FEEDBACK_INFERENCE_PROMPT_MISTAKES_V1 = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing reasoning-based feedback explaining why a relation extraction system arrived at this incorrect prediction.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given an input instance for relation inference below that contains:
- A relation and its description
- A support instance (an example where the relation holds)
- A query sentence
- A ground-truth label (yes/no)
- A yes/no inference made by another LLM on this instance

The ground-truth label is provided only as contextual information. The feedback model’s task is to explain the most likely reasoning process that led to the model’s incorrect yes/no decision, highlighting the misunderstanding, missing evidence, or heuristic that caused the error.

Instance:
```
Relation: #RELATION#
Relation Description: #RELATION_DESCRIPTION#
Support Instance: #SUPPORT_INSTANCE#

Query: #QUERY#

Label: #LABEL#
LLM Inference: #INFERENCE#
```

You may perform reasoning internally, but provide only your final remarks or official feedback within the <f> and </f> tags.
'''

MUTATION_PROMPT_V1 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising and improving prompts based on feedback from previous model predictions.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below a prompt that is used by another LLM to make an inference for the task:
```
#INFERENCE_PROMPT#
```

Using this prompt, another LLM was tested on three instances of the task. Below, you are given the inputs, the inference made by the other LLM, and feedback for each task.
```
Task 1
Relation: #RELATION_1#
Relation Description: #RELATION_DESCRIPTION_1#
Support Instance: #SUPPORT_INSTANCE_1#
Query Sentence: #QUERY_1#
Ground-Truth Label: #LABEL_1#
LLM Inference: #INFERENCE_1#
Feedback: #FEEDBACK_1#

Task 2
Relation: #RELATION_2#
Relation Description: #RELATION_DESCRIPTION_2#
Support Instance: #SUPPORT_INSTANCE_2#
Query Sentence: #QUERY_2#
Ground-Truth Label: #LABEL_2#
LLM Inference: #INFERENCE_2#
Feedback: #FEEDBACK_2#

Task 3
Relation: #RELATION_3#
Relation Description: #RELATION_DESCRIPTION_3#
Support Instance: #SUPPORT_INSTANCE_3#
Query Sentence: #QUERY_3#
Ground-Truth Label: #LABEL_3#
LLM Inference: #INFERENCE_3#
Feedback: #FEEDBACK_3#
```

Carefully read the inputs, outputs, and feedback to identify problems with the current prompt. 
Your task is to generate a revised form of the prompt so that the other LLM can improve the model’s generalization when using the prompt. 
You may modify, add to, or remove any instructions or content in the current prompt in order to improve the prediction and enhance generalization.

Do not change any placeholder tokens enclosed in # (e.g., #LIST_OF_PLACEHOLDERS#). These placeholders must remain exactly the same. Only the surrounding instructional text may be revised.

You may perform reasoning internally, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

# updated from V1 - this is for new format of prompts (Instruction + Examples + Input prompt) - e.g. removed the requirement of having placeholders
MUTATION_PROMPT_V2 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising and improving prompts based on feedback from previous model predictions.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below a prompt that is used by another LLM to make an inference for the task:
```
#INFERENCE_PROMPT#
```

Using this prompt, another LLM was tested on three instances of the task. Below, you are given the inputs, the inference made by the other LLM, and feedback for each task.
```
Task 1
Relation: #RELATION_1#
Relation Description: #RELATION_DESCRIPTION_1#
Support Instance: #SUPPORT_INSTANCE_1#
Query Sentence: #QUERY_1#
Ground-Truth Label: #LABEL_1#
LLM Inference: #INFERENCE_1#
Feedback: #FEEDBACK_1#

Task 2
Relation: #RELATION_2#
Relation Description: #RELATION_DESCRIPTION_2#
Support Instance: #SUPPORT_INSTANCE_2#
Query Sentence: #QUERY_2#
Ground-Truth Label: #LABEL_2#
LLM Inference: #INFERENCE_2#
Feedback: #FEEDBACK_2#

Task 3
Relation: #RELATION_3#
Relation Description: #RELATION_DESCRIPTION_3#
Support Instance: #SUPPORT_INSTANCE_3#
Query Sentence: #QUERY_3#
Ground-Truth Label: #LABEL_3#
LLM Inference: #INFERENCE_3#
Feedback: #FEEDBACK_3#
```

Carefully read the inputs, outputs, and feedback to identify problems with the current prompt. 
Your task is to generate a revised form of the prompt so that the other LLM can improve the model’s generalization when using the prompt. 
You may modify, add to, or remove any instructions or content in the current prompt in order to improve the prediction and enhance generalization.

You may perform reasoning internally, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

MUTATION_NO_FEEDBACK_PROMPT_V1 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising prompts to improve generalization.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below a prompt that is used by another LLM to make an inference for the task:
```
#INFERENCE_PROMPT#
```

Carefully read the prompt. Your task is to generate a revised form of the prompt by making modifications so that the other LLM can improve its generalization when using the prompt.
You may modify, add, remove, or rephrase any word, phrase, or content in the current prompt to enhance generalization.

Do not change any placeholder tokens enclosed in # (e.g., #LIST_OF_PLACEHOLDERS#). These placeholders must remain exactly the same. Only the surrounding instructional text may be revised.

You may perform reasoning internally, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

# updated from V1 - this is for new format of prompts (Instruction + Examples + Input prompt) - e.g. removed the requirement of having placeholders
MUTATION_NO_FEEDBACK_PROMPT_V2 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising prompts to improve generalization.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below a prompt that is used by another LLM to make an inference for the task:
```
#INFERENCE_PROMPT#
```

Carefully read the prompt. Your task is to generate a revised form of the prompt by making modifications so that the other LLM can improve its generalization when using the prompt.
You may modify, add, remove, or rephrase any word, phrase, or content in the current prompt to enhance generalization.

You may perform reasoning internally, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

MUTATION_RANDOM_PROMPT_V1 = '''Make random changes to the text below. You may add, remove, replace, or edit any part of the text.

```
#INFERENCE_PROMPT#
```

Do not change any placeholder tokens enclosed in # (e.g., #LIST_OF_PLACEHOLDERS#). These placeholders must remain exactly the same.
Output the edited text enclosed within the <p> and </p> tags.
'''

# updated from V1 - this is for new format of prompts (Instruction + Examples + Input prompt) - e.g. removed the requirement of having placeholders
MUTATION_RANDOM_PROMPT_V3 = '''Make random changes to the text below. You may add, remove, replace, or edit any part of the text.

```
#INFERENCE_PROMPT#
```

Output the edited text enclosed within the <p> and </p> tags.
'''

# corrected random prompt from V1 - "random" term removed
MUTATION_RANDOM_PROMPT_V2 = '''Make changes to the text below. You may add, remove, replace, or edit any part of the text.

```
#INFERENCE_PROMPT#
```

Do not change any placeholder tokens enclosed in # (e.g., #LIST_OF_PLACEHOLDERS#). These placeholders must remain exactly the same.
Output the edited text enclosed within the <p> and </p> tags.
'''

# updated from V2 - this is for new format of prompts (Instruction + Examples + Input prompt) - e.g. removed the requirement of having placeholders
MUTATION_RANDOM_PROMPT_V4 = '''Make changes to the text below. You may add, remove, replace, or edit any part of the text.

```
#INFERENCE_PROMPT#
```

Output the edited text enclosed within the <p> and </p> tags.
'''

INFERENCE_PROMPT_V1 = f'''You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with subject and object tags, respectively. You need to decide whether the relation holds between the Subject and the Object in the query sentence.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

If the relation holds between the Subject and Object in the query sentence, say "yes"; otherwise, say "no". Output only "yes" or "no", and nothing else.
'''

INFERENCE_PROMPT_PLACEHODERS_V1 = ["#RELATION#", "#RELATION_DESCRIPTION#", "#QUERY_SENTENCE#", "#SUPPORT_SENTENCE_BLOCK#"]

INFERENCE_INSTRUCTION_PROMPT_V1 = f'''You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with subject and object tags, respectively. You need to decide whether the relation holds between the Subject and the Object in the query sentence.

If the relation holds between the Subject and Object in the query sentence, say "yes"; otherwise, say "no". Output only "yes" or "no", and nothing else.
'''

INFERENCE_EXAMPLE_PROMPT_V1 = f'''
Example:

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Answer: #BINARY_ANSWER#
'''

INFERENCE_INPUT_PROMPT_V1 = f'''
Input:

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Answer: '''


EXAMPLE_GENERATION_PROMPT_V1 = '''You are given a relation name, the description of the relation, and a support sentence that exemplifies the relation.

A relation connects two entities in a sentence: the Subject and the Object. The Subject and the Object are indicated with the <subject>...</subject> and <object>...</object> tags, respectively.

Relation name: "#RELATION#"
Relation description: "#RELATION_DESCRIPTION#"
Support Sentence: #SUPPORT_SENTENCE#

Your task is to generate N completely different new example sentences that hold the same relation. You must follow these guidelines:
- In each example, include subject and object tags to identify the Subject and the Object entities.
- To increase diversity, use different words, phrases, and sentence structures across different examples.

Output in the following format.

01: your 1st example sentence
02: your 2nd example sentence
...
N: your Nth example sentence
'''


FEEDBACK_PROMPT_MAP = {
    "correct_and_mistakes_v1": FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1,
    "correct_v1": FEEDBACK_INFERENCE_PROMPT_CORRECT_V1,
    "mistakes_v1": FEEDBACK_INFERENCE_PROMPT_MISTAKES_V1,
}

MUTATION_PROMPT_MAP = {
    "v1": MUTATION_PROMPT_V1,
    "random_v1": MUTATION_RANDOM_PROMPT_V1,
    "random_v2": MUTATION_RANDOM_PROMPT_V2,
    "no_feedback_v1": MUTATION_NO_FEEDBACK_PROMPT_V1
}

MUTATION_PROMPT_MAP_V2 = {
    "v1": MUTATION_PROMPT_V2,
    "random_v1": MUTATION_RANDOM_PROMPT_V3,
    "random_v2": MUTATION_RANDOM_PROMPT_V4,
    "no_feedback_v1": MUTATION_NO_FEEDBACK_PROMPT_V2,
}

INFERENCE_MODE_SEPARATE_NO_EXAMPLES = "separate_no_examples"
INFERENCE_MODE_SEPARATE_WITH_EXAMPLES = "separate_with_examples"
INFERENCE_MODE_NON_SEPARATE = "non_separate"
INFERENCE_MODE_CHOICES = (
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    INFERENCE_MODE_SEPARATE_WITH_EXAMPLES,
    INFERENCE_MODE_NON_SEPARATE,
)


def resolve_mutation_prompt(prompt_key: str, inference_mode: str) -> str:
    if inference_mode == INFERENCE_MODE_NON_SEPARATE:
        return MUTATION_PROMPT_MAP[prompt_key]
    if inference_mode in (
        INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
        INFERENCE_MODE_SEPARATE_WITH_EXAMPLES,
    ):
        return MUTATION_PROMPT_MAP_V2[prompt_key]
    raise ValueError(f"Unsupported inference_mode: {inference_mode}")


def compose_inference_prompt(
    *,
    inference_mode: str,
    inference_prompt: str,
    inference_instruction_prompt: str = "",
    inference_example_prompt: str = "",
    inference_input_prompt: str = "",
    relation: str,
    relation_description: str,
    support_block: str,
    query_sentence: str,
    example_query_sentence: str,
) -> str:
    if inference_mode == INFERENCE_MODE_NON_SEPARATE:
        prompt = inference_prompt or INFERENCE_PROMPT_V1
        prompt = prompt.replace("#RELATION#", relation)
        prompt = prompt.replace("#RELATION_DESCRIPTION#", relation_description)
        prompt = prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
        prompt = prompt.replace("#QUERY_SENTENCE#", query_sentence)
        return prompt

    if inference_mode == INFERENCE_MODE_SEPARATE_NO_EXAMPLES:
        instruction_prompt = (
            inference_instruction_prompt or inference_prompt or INFERENCE_INSTRUCTION_PROMPT_V1
        )
        input_prompt = inference_input_prompt or INFERENCE_INPUT_PROMPT_V1
        input_prompt = input_prompt.replace("#RELATION#", relation)
        input_prompt = input_prompt.replace("#RELATION_DESCRIPTION#", relation_description)
        input_prompt = input_prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
        input_prompt = input_prompt.replace("#QUERY_SENTENCE#", query_sentence)
        return instruction_prompt + input_prompt

    # todo: this will be iterative i.e. K examples
    if inference_mode == INFERENCE_MODE_SEPARATE_WITH_EXAMPLES:
        instruction_prompt = (
            inference_instruction_prompt or inference_prompt or INFERENCE_INSTRUCTION_PROMPT_V1
        )
        example_prompt = inference_example_prompt or INFERENCE_EXAMPLE_PROMPT_V1
        example_prompt = example_prompt.replace("#RELATION#", relation)
        example_prompt = example_prompt.replace("#RELATION_DESCRIPTION#", relation_description)
        example_prompt = example_prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
        example_prompt = example_prompt.replace("#QUERY_SENTENCE#", example_query_sentence)
        example_prompt = example_prompt.replace("#BINARY_ANSWER#", "yes")

        input_prompt = inference_input_prompt or INFERENCE_INPUT_PROMPT_V1
        input_prompt = input_prompt.replace("#RELATION#", relation)
        input_prompt = input_prompt.replace("#RELATION_DESCRIPTION#", relation_description)
        input_prompt = input_prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
        input_prompt = input_prompt.replace("#QUERY_SENTENCE#", query_sentence)

        return instruction_prompt + example_prompt + input_prompt

    raise ValueError(f"Unsupported inference_mode: {inference_mode}")


def apply_tag_overrides(
    prompt: str,
    *,
    feedback_open_tag: str,
    feedback_close_tag: str,
    prompt_open_tag: str,
    prompt_close_tag: str,
) -> str:
    if feedback_open_tag and feedback_close_tag:
        prompt = prompt.replace("<f>", feedback_open_tag).replace("</f>", feedback_close_tag)
    if prompt_open_tag and prompt_close_tag:
        prompt = prompt.replace("<p>", prompt_open_tag).replace("</p>", prompt_close_tag)
    return prompt
