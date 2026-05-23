python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:2 --ep_start 0 --ep_end 150000 --batch_size 4 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_rpo_node_y_greater-tg --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. Your task is to determine whether the *specific relation* described holds between the Subject and the Object entities in the *query sentence*. Carefully examine the query sentence and consider the relationship between the subject and object entities. Do not answer "yes" simply because the subject and object are related in a general sense; verify that the query sentence explicitly or implicitly expresses the specified relation. Base your decision *solely* on the content of the query sentence. Ignore the support sentence.

When evaluating whether a relation holds, focus on the *core meaning* of the relation. Consider that the query sentence may express the relation using different phrasing or terminology than the relation description. Look for synonyms and related concepts that indicate the presence of the relation. Pay attention to context clues and implied relationships; the relation does not always need to be stated explicitly. Look beyond surface-level phrasing.

Avoid being distracted by surrounding details or circumstantial information within the query sentence that do not directly establish the relation. Prioritize understanding the core relationship between the subject and object, even if it is not stated using the exact words from the relation description. Consider the overall meaning of the sentence, even if the relationship is expressed indirectly.

<b>To help you reason, consider these questions before answering:</b>
* Does the sentence state or strongly imply a connection between the subject and the object that aligns with the relation description?
* Are there keywords or phrases that are closely associated with the relation, even if they are not exact synonyms?
* Could a reasonable person, knowing the relation description, conclude that the subject and object are connected in this way based *only* on the query sentence?
* **Specifically, identify the words or phrases in the query sentence that relate to the subject and object and explain how they suggest the relation. If the relation is not explicitly stated, what is the implied connection?**

<b>Pay close attention to the entities themselves. Are you focusing on the correct entities as the subject and object? Confirm you'"'"'ve correctly identified which entity corresponds to which role.</b>

<b>Be mindful of potential ambiguities. Does the sentence present multiple possible interpretations? If so, choose the interpretation that best supports the relation.</b>

<b>Do not let pronouns or other linguistic complexities obscure the core meaning of the sentence. Focus on the concrete information presented.**

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"! otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:2 --ep_start 0 --ep_end 150000 --batch_size 4 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_rpo_node_y_gradpo-gen --prompt 'You are given a relation name, a description of the relation in brackets, a example exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. Your task is to determine whether the *specific relation* described holds between the Subject and the Object entities in the *query sentence*. Carefully examine the query sentence and consider the relationship between the subject and object entities. Do not answer "yes" simply because the subject and object are related in a general sense; verify that the query sentence explicitly or implicitly expresses the specified relation. Base your decision *solely* on the content of the query sentence. Ignore the example.

When evaluating whether a relation holds, focus on the *core meaning* of the relation. Consider that the query sentence may express the relation using different phrasing or terminology than the relation description. Look for synonyms and related concepts that indicate the presence of the relation. Pay attention to context clues and implied relationships; the relation does not always need to be stated explicitly. Look beyond surface-level phrasing.

Avoid being distracted by surrounding details or circumstantial information within the query sentence that do not directly establish the relation. Prioritize understanding the core relationship between the subject and object, even if it is not stated using the exact words from the relation description. Consider the overall meaning of the sentence, even if the relationship is expressed indirectly.

To help you reason, consider these questions before answering:
* Does the sentence state or strongly imply a connection between the subject and the object that aligns with the relation description?
* Are there keywords or phrases that are closely associated with the relation, even if they are not exact synonyms?
* Could a reasonable person, knowing the relation description, conclude that the subject and object are connected in this way based *only* on the query sentence?
* **Specifically, identify the words or phrases in the query sentence that relate to the subject and object and explain how they suggest the relation. If the relation is not explicitly stated, what is the implied connection?**

Pay close attention to the entities themselves. Are you focusing on the correct entities as the subject and object? Confirm you'"'"'ve correctly identified which entity corresponds to which role.

Be mindful of potential ambiguities. Does the sentence present multiple possible interpretations? If so, choose the interpretation that best supports the relation.

Do not let pronouns or other linguistic complexities obscure the core meaning of the sentence. Focus on the concrete information presented.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'