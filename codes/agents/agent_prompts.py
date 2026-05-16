FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1 = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing feedback explaining why a relation extraction system arrived at a particular yes/no decision, for both correct and incorrect predictions.

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

Please reason through the problem, but provide your final feedback only within the <f> and </f> tags.
'''

FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1_GEMMA = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing feedback explaining why a relation extraction system arrived at a particular yes/no decision, for both correct and incorrect predictions.

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

Please reason through the problem first, and then provide your final feedback enclosed within the <f> and </f> tags.
'''

# original
# Please reason through the problem, but provide your final feedback only within the <f> and </f> tags.

# gemma special
# Please reason through the problem first, and then provide your final feedback enclosed within the <f> and </f> tags.

FEEDBACK_INFERENCE_PROMPT_CORRECT_V1 = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing feedback explaining why a relation extraction system arrived at this correct prediction.

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

Please reason through the problem, but provide your final feedback only within the <f> and </f> tags.
'''

FEEDBACK_INFERENCE_PROMPT_MISTAKES_V1 = '''You are an expert feedback model for a relation extraction inference task. Specifically, you are skilled at providing feedback explaining why a relation extraction system arrived at this incorrect prediction.

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

Please reason through the problem, but provide your final feedback only within the <f> and </f> tags.
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

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

# todo: "change the term TASK to test input"
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
Your task is to generate a revised version of the prompt that helps the other LLM generalize better when using it.
You may modify, add to, or remove any instructions or content in the current prompt in order to improve the prediction and enhance generalization.

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

MUTATION_PROMPT_V2_GEMMA = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising and improving prompts based on feedback from previous model predictions.

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

Analyze the inputs, outputs, and feedback to identify prompt weaknesses.
Revise the prompt with proper instructions so that it improves.

Please reason through the problem first, and then output the revised prompt enclosed within the <p> and </p> tags.
'''

# Analyze the inputs, outputs, and feedback to identify prompt weaknesses, then revise the prompt so it generalizes better.

# Please reason through the problem first, and then output the revised prompt enclosed within the <p> and </p> tags.


# original
# Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.

# gemma special
# Use the relation name and description as the main definition; treat the support sentence as only one example, not a template the query must closely match.

# Please reason through the problem first, and then output the revised prompt enclosed within the <p> and </p> tags.

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

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
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

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

GRADIENT_REGION_MUTATION_PROMPT_V1 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in making precise, local prompt edits that improve binary yes/no relation inference behavior while preserving the rest of the prompt.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below the current instruction prompt used by another LLM for this task. The editable regions are marked using numbered tags such as <edit_start_1> and <edit_end_1>.
These numbered edit regions are ordered by gradient ranking, where `edit_region_1` is the highest-ranked region, `edit_region_2` is the second-highest-ranked region, and so on:
```
#MARKED_PROMPT#
```

Your task is to generate one revised full instruction prompt by editing only the marked `edit_region_X` regions so that the prompt can improve generalization for this relation extraction task.
Keep the rest of the prompt unchanged as much as possible.
You may make a minimal nearby adjustment only if needed to keep the revised prompt natural and semantically coherent.
Do not include any edit tags such as <edit_start_1> or <edit_end_1> in the final revised prompt.

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

GRADIENT_REGION_CANDIDATE_SUGGESTION_PROMPT_V1 = '''You are an expert on suggesting replacements for targeted spans in a prompt.

The editable spans are marked using tags such as <span_1>...</span_1>, <span_2>...</span_2>, and so on.

Input Prompt with editable spans:
```
#MARKED_PROMPT#
```

Editable spans:
#REGION_CANDIDATE_REQUEST_BLOCKS#

Task:
Suggest #NUM_CANDIDATES# replacements for each editable span so that every replacement fits naturally in context.

Rules:
- Preserve the meaning and role of each editable span
- Treat each tagged span as a single unit
- Return replacements only for the editable spans
- A candidate may be identical to the original span text if keeping it unchanged is the best option

Please reason through the problem, but output the replacements in JSON:
```json
{
  "span_1": {
    "candidates": [
      "replacement 1",
      "...",
      "replacement #NUM_CANDIDATES#"
    ]
  },
  ...
  "span_#NUM_REGIONS#": {
    "candidates": [
      "replacement 1",
      "...",
      "replacement #NUM_CANDIDATES#"
    ]
  }
}
```
'''

GRADIENT_REGION_CANDIDATE_COMBINATION_PROMPT_V1 = '''You are an expert in selecting effective candidate combinations for span replacements in a relation extraction prompt.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given the current instruction prompt with targeted spans below:
```
#ALL_MARKED_PROMPT#
```
Each editable span is marked with tags such as <span_1>...</span_1>, <span_2>...</span_2>, and so on.

The editable spans are listed below together with candidate replacements for each span:
#REGION_CANDIDATE_BLOCKS#

Your task is to generate #NUM_PROMPT# diverse and promising candidate combinations of edits for the editable spans.
For each span, select exactly one candidate replacement or you may leave a span unchanged if that is better. 
Each combination must be distinct and diverse.

Output only the combinations in the following JSON format:
```json
{
  "combination_1": {
        "span_1": "selected replacement",
        ...
        "span_#NUM_REGIONS#": "selected replacement"
      },
  ...
  "combination_#NUM_PROMPT#": {
        "span_1": "selected replacement",
        ...
        "span_#NUM_REGIONS#": "selected replacement"
      }
}
```
'''

GRADIENT_REGION_CANDIDATE_SYNTHESIS_PROMPT_V1 = '''You are an expert prompt generator for a relation extraction inference task.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object entities.

You are given the current instruction prompt with targeted spans below:
```
#ALL_MARKED_PROMPT#
```
Each editable span is marked with tags such as <span_1>...</span_1>, <span_2>...</span_2>, and so on.

Replacements for each spans are given below:
#SELECTED_REPLACEMENTS#

Your task is to generate a revised instruction prompt by applying the given replacements to the corresponding spans.
Use the replacements exactly as provided, except for minimal local adjustments if necessary for spelling and grammatical correctness or coherence.
Do not modify any other parts of the prompt (but remove the span tags).

*** Remove the span tags (e.g., <span_1>...</span_1>, <span_2>...</span_2>, and so on.) from the revised prompt. ***

Output only the revised prompt.
'''

GRADIENT_REGION_CANDIDATE_SYNTHESIS_PROMPT_SINGLE_REGION_V1 = '''You are an expert prompt generator for a relation extraction inference task.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object entities.

You are given the current instruction prompt with a targeted span below:
```
#MARKED_PROMPT#
```

Replacement for the target span is given below:
#SELECTED_REPLACEMENT#

Your task is to generate a revised instruction prompt by applying the given replacement to the targeted span.
Use the replacement exactly as provided, except for minimal local adjustments if necessary for spelling and grammatical correctness or coherence.
Do not modify any other parts of the prompt (but remove the target span tags).

*** Remove the target span tags (<target>...</target>) from the revised prompt. ***

Output only the revised prompt.
'''

LPO_LOCATION_TAGGING_PROMPT_V1 = '''A relation extraction prompt helps an LLM decide whether a query sentence expresses a target relation between the subject and object entities, using a support sentence for that relation. The classifier must answer with exactly one token: "yes" or "no".

Current prompt:
```
#INFERENCE_PROMPT#
```

Feedback examples from the current prompt:
#FEEDBACK_EXAMPLES#

Let's think step by step.
First, identify the scope of tokens within the prompt where edits should take place.
Prompt edits include adding, deleting or modifying tokens.
Mark the scope of the prompt that needs editing by putting <edit>, </edit> tags.
You can have multiple <edit> tags and each <edit> tag should not entail more than #MAX_WORDS_PER_EDIT_TAG# words.
Use at most #MAX_EDIT_TAGS# <edit> tags.
Do not cover the whole sentence with multiple <edit> tags.
Reply with the prompt with <edit>, </edit> tags.
Do not include any other text.
'''

LPO_LOCAL_REWRITE_PROMPT_V1 = '''A relation extraction prompt helps an LLM decide whether a query sentence expresses a target relation between the subject and object entities, using a support sentence for that relation. The classifier must answer with exactly one token: "yes" or "no".

Current prompt with local edit scopes:
```
#TAGGED_PROMPT#
```

Feedback examples from the current prompt:
#FEEDBACK_EXAMPLES#

Generate one revised full prompt by editing only the text inside the <edit>, </edit> tags.
You may add, delete, or modify tokens inside those local scopes.
Keep the rest of the prompt unchanged except for minimal grammar cleanup at the edit boundaries.
Do not include <edit> or </edit> tags in the revised prompts.

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

# Your task is to generate a revised instruction prompt by applying the given replacements to the corresponding spans.
# Use the replacements exactly as provided.
# Do not modify any other parts of the prompt, except for minimal local adjustments if necessary for grammatical correctness or coherence.
# Do not output span labels in the revised prompt.

# Output only the revised prompt enclosed within the <p> and </p> tags.
# '''

MUTATION_TRACES_PROMPT_V1 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising prompts to improve generalization.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given a short sequence of prompts that were previously used by another LLM to perform inference for this task. 
These prompts represent successive evolutions of the same prompt and may include up to three versions (for example, a root prompt only, or the root prompt with one or two evolved descendants). Each prompt is provided with a validation score.

#INFERENCE_PROMPT_TRACES#

Carefully read the latest prompt and any earlier versions provided. Analyze how the prompt has evolved and how changes may have affected generalization performance, using the validation scores as guidance.

Your task is to generate a revised version of the latest prompt that improves generalization for the relation extraction inference task. You may modify, add, remove, or rephrase any part of the latest prompt to enhance generalization. 
If fewer prompt versions are provided, base your reasoning only on the available information.

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

MUTATION_TRACES_PROMPT_SEGMENT_V1= ''' 
==
Prompt #PROMPT_NUMBER#

```
#PROMPT#
```

Score: #SCORE#
'''

MUTATION_TRACES_DIFFERENCES_PROMPT_V1 = '''You are an expert prompt generator for a relation extraction inference task. You specialize in revising prompts to improve generalization.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given a short sequence of prompts that were previously used by another LLM to perform inference for this task. 
These prompts represent successive evolutions of the same prompt and may include up to three versions (for example, a root prompt only, or the root prompt with one or two evolved descendants). 
The root prompt is provided in full, and each subsequent prompt is provided as the edit operations applied to its immediate parent prompt (e.g., the third prompt contains edits applied to the second prompt). 
Each prompt is provided with a validation score.

#INFERENCE_PROMPT_TRACES#

Carefully read the latest prompt and any earlier versions provided. Analyze how the prompt has evolved and how changes may have affected generalization performance, using the validation scores as guidance.

Your task is to generate a revised version of the latest prompt that improves generalization for the relation extraction inference task. You may modify, add, remove, or rephrase any part of the latest prompt to enhance generalization. 
If fewer prompt versions are provided, base your reasoning only on the available information.

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
'''

MUTATION_TRACES_DIFFERENCES_PROMPT_SEGMENT_V1= ''' 
==
Prompt #PROMPT_NUMBER#

```
#PROMPT_CONTENT#
```

Score: #SCORE#
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

DIFFERENTIATE_PROMPT = """You are an expert prompt differencing agent. You are given a parent prompt and a child prompt. Your task is to output all edit operations needed to transform the parent prompt into the child prompt.

Parent prompt:
```
#PROMPT1#
```

Child prompt:
```
#PROMPT2#
```

Rules:
- Output all changes in top-to-bottom order.
- Describe each change using the smallest meaningful unit and do not restate unchanged text.
- Use short, instruction-style edits.
- One edit operation per line (word, sentence, or paragraph).

Allowed edits:
- Replace "X" with "Y"
- Add "X"
- Remove "X"

Please reason through the problem, but output only the edits as a numbered list enclosed within <d> and </d> tags.

Format:
<d>
1. Replace / Add / Remove operation
...
n. Replace / Add / Remove operation
</d>
"""

CLUSTER_CATEGORY_ASSIGNMENT_PROMPT = """You are an expert at categorizing feedback about model mistakes into reusable error categories. Use an existing category if the feedback fits, or create a new category if none match.

Current feedback:
```
#FEEDBACK#
```

Existing categories:
#CATEGORIES#

Decide whether the feedback belongs to one of the existing categories.
- If it matches an existing category, select that category id.
- Otherwise create a new category with a short name and a concise description.
- Prefer general, reusable categories that capture the underlying mistake.

Please reason through the problem and return the assignment decision in exactly this format:
<assignment>
decision: existing OR new
category_id: existing id, or NEW
name: short category name
description: concise category description
</assignment>
"""

CLUSTER_MUTATION_PROMPT_V1 = """You are an expert prompt generator for a relation extraction inference task. You specialize in revising prompts to improve generalization based on grouped error categories.

A relation captures the connection between two entities in a sentence by describing their relationship. We will refer to these entities as the subject and object entities.
The task requires inferring a binary (yes/no) answer based on whether the query sentence expresses this relation between the subject and the object.

You are given the current prompt:
```
#INFERENCE_PROMPT#
```

The following feedback categories summarize recurring mistakes made with this prompt:
#CATEGORY_BLOCK#

Each category describes a type of systematic error observed when the prompt is used.
Carefully review the categories to identify weaknesses in the current prompt. 
Your task is to generate a revised form of the prompt so that another LLM using it can avoid these error patterns and improve generalization.
You may modify, add to, or remove any instructions or content in the current prompt in order to address the issues described by the categories.

Please reason through the problem, but output only the revised prompt enclosed within the <p> and </p> tags.
"""

INFERENCE_PROMPT_V1 = f'''You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with subject and object tags, respectively. You need to decide whether the relation holds between the Subject and the Object in the query sentence.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

If the relation holds between the Subject and Object in the query sentence, say "yes"; otherwise, say "no". Output only "yes" or "no", and nothing else.
'''

INFERENCE_PROMPT_PLACEHODERS_V1 = ["#RELATION#", "#RELATION_DESCRIPTION#", "#QUERY_SENTENCE#", "#SUPPORT_SENTENCE_BLOCK#"]

INFERENCE_INSTRUCTION_PROMPT_V1 = f'''You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'''

# INFERENCE_INSTRUCTION_PROMPT_V1 = f'''You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence.

# A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with subject and object tags, respectively. 

# You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.
# '''

INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1 = '''

Output only "yes" or "no" as answer, with no explanation or additional text.
'''

INFERENCE_EXAMPLE_PROMPT_V1 = f'''
==

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Answer: #BINARY_ANSWER#

'''

INFERENCE_INPUT_PROMPT_V1 = f'''
==

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
    "correct_and_mistakes_v1_gemma": FEEDBACK_INFERENCE_PROMPT_CORRECT_AND_MISTAKES_V1_GEMMA,
    "correct_v1": FEEDBACK_INFERENCE_PROMPT_CORRECT_V1,
    "mistakes_v1": FEEDBACK_INFERENCE_PROMPT_MISTAKES_V1,
}

MUTATION_PROMPT_MAP = {
    "v1": MUTATION_PROMPT_V2,
    "v1_gemma": MUTATION_PROMPT_V2_GEMMA,
    "random_v1": MUTATION_RANDOM_PROMPT_V3,
    "random_v2": MUTATION_RANDOM_PROMPT_V4,
    "no_feedback_v1": MUTATION_NO_FEEDBACK_PROMPT_V2,
    "traces_v1": MUTATION_TRACES_PROMPT_V1,
    "traces_differences_v1": MUTATION_TRACES_DIFFERENCES_PROMPT_V1,
}

MUTATION_PROMPT_GROUP_MAP = {
    "group_1": ["no_feedback_v1"],
    "group_2": ["v1"],
    "group_3": ["random_v2"],
    "group_4": ["traces_v1"],
    "group_5": ["no_feedback_v1", "traces_v1"],
    "group_6": ["v1", "traces_v1"],
    "group_7": ["random_v2", "traces_v1"],
    "group_8": ["traces_differences_v1"],
    "group_9": ["no_feedback_v1", "traces_differences_v1"],
    "group_10": ["v1", "traces_differences_v1"],
    "group_gemma": ["v1_gemma"],
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
    if inference_mode in INFERENCE_MODE_CHOICES:
        return MUTATION_PROMPT_MAP[prompt_key]
    raise ValueError(f"Unsupported inference_mode: {inference_mode}")


def resolve_mutation_prompts_from_group(group_id: str, inference_mode: str) -> list[str]:
    if inference_mode not in INFERENCE_MODE_CHOICES:
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")
    keys = MUTATION_PROMPT_GROUP_MAP[group_id]
    return [MUTATION_PROMPT_MAP[key] for key in keys]


def compose_inference_prompt(
    *,
    inference_mode: str,
    inference_prompt: str,
    inference_instruction_prompt: str = "",
    inference_answer_instruction_prompt: str = "",
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
            (inference_instruction_prompt or INFERENCE_INSTRUCTION_PROMPT_V1)
            + (
                inference_answer_instruction_prompt
                or INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1
            )
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
            (inference_instruction_prompt or INFERENCE_INSTRUCTION_PROMPT_V1)
            + (
                inference_answer_instruction_prompt
                or INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1
            )
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
