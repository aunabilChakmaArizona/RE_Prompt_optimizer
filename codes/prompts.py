EVALUATE_INFERENCE_PROMPT_V1 = '''You are an good evaluator for a relation extraction inference task.

In this task, a relation defines the connection between two selected entities in a sentence: a subject and an object.
The relation describes the role or relationship that the object has with respect to the subject.
The task is to infer a yes or no answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below an input instance for relation inference that contains:
- A relation and its description
- A support instance (an example where the relation holds)
- A query sentence
- A ground-truth label (yes/no)
- A yes/no inference made by another LLM

Your task is to explain the likely reasoning that led the other LLM to predict yes or no, using only the given relation description, support instance, and query. 
Do not re-evaluate the relation yourself. Instead, explain the LLM’s decision in light of the ground-truth label.
- If the prediction matches the label, explain what cues or evidence likely led to that choice.
- If it does not match, explain what misunderstanding, missing evidence, or heuristic likely caused the prediction.

Instance:
```
Relation: #RELATION#
Relation description: #RELATION_DESCRIPTION#
Support Instance: #SUPPORT_INSTANCE#

Query: #QUERY#

Label: #LABEL#
LLM inference: #INFERENCE#
```

Output your feedback only inside <f> and </f> tags.
'''

MUTATION_PROMPT_V1 = '''You are an good prompt generator for a relation extraction inference task.

In this task, a relation defines the connection between two selected entities in a sentence: a subject and an object.
The relation describes the role or relationship that the object has with respect to the subject.
The task is to infer a yes or no answer based on whether the query sentence expresses this relation between the subject and the object.

You are given below an prompt that is used by the another LLM to make the inference of the task:
```
#INFERENCE_PROMPT#
```

Using this prompt, the other LLM was tested on 3 instances of this task. Below you are given for each of the tasks: the inputs, the inference made by the other LLM, and a feedback.
```
Task 1
Relation: #RELATION_1#
Relation Description: #RELATION_DESCRIPTION_1#
Support Instance: #SUPPORT_INSTANCE_1#
Query sentence: #QUERY_1#
Ground-truth label: #LABEL_1#
LLM infernence: #QUERY_1#
Feedback: #FEEDBACK_1#

Task 2
Relation: #RELATION_2#
Relation Description: #RELATION_DESCRIPTION_2#
Support Instance: #SUPPORT_INSTANCE_2#
Query sentence: #QUERY_2#
Ground-truth label: #LABEL_2#
LLM infernence: #QUERY_2#
Feedback: #FEEDBACK_2#

Task 3
Relation: #RELATION_3#
Relation Description: #RELATION_DESCRIPTION_3#
Support Instance: #SUPPORT_INSTANCE_3#
Query sentence: #QUERY_3#
Ground-truth label: #LABEL_3#
LLM infernence: #QUERY_3#
Feedback: #FEEDBACK_3#
```

Read the inputs, outputs, and the feedbacks carefully. Idenfity the problems with the current given prompt.
Your task is to generate a revised form of the prompt so that the other LLM can improve the generalization of the task while using it. 
You may include modify, add or remove any instructions provided in the current prompt. 
You can put additional details or do anything in the prompt that may help to improve the prediction with generalization.

Output the revised prompt only under <p> and </p> tags.
'''