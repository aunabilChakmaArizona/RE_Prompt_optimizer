python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_rpo_node_x_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence. A relation connects the Subject and Object entities, which are indicated with the subject and object tags, respectively. 

To determine if the relation holds between the Subject and Object entities in the query sentence, strictly follow these steps:
1. **Verify the subject and object types**: 
   - For relations starting with "org:", the **subject must be an organization** (not a person or role). 
   - The **object must align with the relation'"'"'s definition** (e.g., "org:subsidiaries" requires a subsidiary entity, not a role or category; "org:political/religious_affiliation" accepts a religious or political category, not a person). 
2. **Check the relationship**: Confirm that the query sentence explicitly links the Subject and Object as per the relation'"'"'s description. For example, "org:subsidiaries" requires a hierarchical connection between an organization and its subsidiary (e.g., "Company A owns Company B"), not a personal or abstract relationship. 
3. **Recognize object types**: 
   - The object can be a **specific entity** (e.g., "US-based Ryder Public Transportation Services") or a **category** (e.g., "Christian", "Islamist") if the relation explicitly allows it. 
   - Avoid treating abstract terms (e.g., "Christian") as roles or people unless the relation'"'"'s description explicitly permits it. 
4. **Focus on explicit connections**: Only answer "yes" if the query sentence directly establishes the relation between the subject and object as defined. For example, "org:number_of_employees/members" requires a numerical value (e.g., "42,000") to indicate the count, not a vague term like "many." 

If the relation holds, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_rpo_node_x_gradpo-gen_cross --prompt 'You are given a relation name, a definition of the relation in bracketed text, a support example sentence exemplifying the relation, and a query inquiry sentence. A relation connects the Subject and Object entities, which are indicated with the subject and object tags, respectively. 

To determine if the relation holds between the Subject and Object entities in the query sentence, strictly follow these steps:
1. **Verify the subject and object types**: 
   - For relations starting with "org:", the **subject must be an organization** (not a person or role). 
   - The **object must align with the relation'"'"'s definition** (e.g., "org:subsidiaries" requires a subsidiary entity, not a role or category; "org:political/religious_affiliation" accepts a religious or political category, not a person). 
2. **Check the relationship**: Confirm that the query sentence explicitly links the Subject and Object as per the relation'"'"'s description. For example, "org:subsidiaries" requires a hierarchical connection between an organization and its subsidiary (e.g., "Company A owns Company B"), not a personal or abstract relationship. 
3. **Recognize object types**: 
   - The object can be a **specific entity** (e.g., "US-based Ryder Public Transportation Services") or a **category** (e.g., "Christian", "Islamist") if the relation explicitly allows it. 
   - Avoid treating abstract terms (e.g., "Christian") as roles or people unless the relation'"'"'s definition explicitly permits it. 
4. **Focus on explicit connections**: Only answer "yes" if the query sentence directly establishes the relation between the subject and object as defined. For example, "org:number_of_employees/members" requires a numerical value (e.g., "42,000") to indicate the count, not a vague term like "many." 

If the relation holds, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_rpo_node_y_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence. A relation connects the Subject and Object entities, which are indicated with the subject and object tags, respectively. 

To determine if the relation holds between the Subject and Object entities in the query sentence, strictly follow these steps:
1. **Verify the subject and object types**: The subject must be the entity type expected by the relation (e.g., "org" for organization, "per" for person). The object must match the relation'"'"'s definition (e.g., "org:stateorprovince_of_headquarters" requires the object to be a state/province, not a city or location). If the subject is a location (e.g., "Kennedy Space Center"), it must be explicitly tied to an organization (e.g., NASA) as its headquarters. 
2. **Check the relationship**: Confirm that the query sentence explicitly links the Subject and Object as per the relation'"'"'s description. For example, "org:stateorprovince_of_headquarters" requires the subject to be an organization (e.g., "NASA") and the object to be a state/province (e.g., "Florida"), not a location (e.g., "Kennedy Space Center"). For "org:political/religious_affiliation," the object must be the organization'"'"'s actual political or religious affiliation (e.g., "Islamist"), not a role, descriptor, or faction. 
3. **Avoid role vs. person confusion**: If the object is a role (e.g., "chief executive") or title (e.g., "Rep."), it does not satisfy relations requiring a person (e.g., "employee_of"). Similarly, ensure the subject is the correct entity type (e.g., "org" for organizations, "per" for people). For "org:stateorprovince_of_headquarters," the subject must be an organization (e.g., "NASA"), not a location (e.g., "Kennedy Space Center"), unless the sentence explicitly states the location is the organization'"'"'s headquarters. 
4. **Focus on explicit connections**: Only answer "yes" if the query sentence directly establishes the relation between the subject and object as defined. This includes ensuring the object is the correct entity type (e.g., "state/province" for headquarters, "political/religious affiliation" for org:political/religious_affiliation) and that the subject is the correct entity type for the relation. 

If the relation holds, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_rpo_node_y_gradpo-gen_cross --prompt 'You are given a relation property, a description of the relation in curly braces, a support sentence exemplifying the relation, and a query sentence. A relation connects the Subject and Object entities, which are indicated with the subject and object tags, respectively. 

To determine if the relation holds between the Subject and Object entities in the query sentence, strictly follow these steps:
1. **Verify the subject and object types**: The subject must be the entity type expected by the relation (e.g., "org" for organization, "per" for person). The object must match the relation'"'"'s definition (e.g., "org:stateorarea_of_headquarters" requires the object to be an area, not a city or location). If the subject is a location (e.g., "Kennedy Space Center"), it must be explicitly tied to an organization (e.g., NASA) as its headquarters. 
2. **Check the relationship**: Confirm that the query sentence explicitly links the Subject and Object as per the relation'"'"'s description. For example, "org:stateorarea_of_headquarters" requires the subject to be an organization (e.g., "NASA") and the object to be an area (e.g., "Florida"), not a location (e.g., "Kennedy Space Center"). For "org:political/religious_affiliation," the object must be the organization'"'"'s actual political or religious affiliation (e.g., "Islamist"), not a role, descriptor, or faction. 
3. **Avoid role vs. person confusion**: If the object is a role (e.g., "chief executive") or title (e.g., "Rep."), it does not satisfy relations requiring a person (e.g., "employee_of"). Similarly, ensure the subject is the correct entity type (e.g., "org" for organizations, "per" for people). For "org:stateorarea_of_headquarters," the subject must be an organization (e.g., "NASA"), not a location (e.g., "Kennedy Space Center"), unless the sentence explicitly states the location is the organization'"'"'s headquarters. 
4. **Focus on explicit connections**: Only confirm if the query sentence directly establishes the relation between the subject and object as defined. This includes ensuring the object is the correct entity type (e.g., "area" for headquarters, "political/religious affiliation" for org:political/religious_affiliation) and that the subject is the correct entity type for the relation. 

If the relation holds, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_qwen_rpo_node_x_cross --prompt 'You are given a relation name, a description of the relation in brackets, and a query sentence.

Your task is to determine whether the *specific relation* described holds between the Subject and the Object entities in the *query sentence*. The Subject and the Object entities are indicated with the subject and object tags, respectively. Carefully examine the query sentence, paying close attention to keywords and phrases commonly associated with the specified relation.

Consider the following:

*   **Explicit Statements:** Look for direct statements that express the relation (e.g., "is a subsidiary of," "was born in").
*   **Strong Indicators:** Recognize that relations can be implied or indicated through related language. For example, "native of" often implies "city of birth," and "subsidiary of" directly indicates an organizational relationship.
*   **Contextual Clues:** Evaluate the surrounding context to confirm the relationship.

Do not answer "yes" unless the query sentence explicitly or strongly implies the specified relation. Base your decision *solely* on the content of the query sentence. Ignore any support examples provided.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_qwen_rpo_node_x_gradpo-gen_cross --prompt 'You are given a relation name, a description of the relation in brackets, and a query sentence.

Your task is to determine whether the *specific relation* described holds between the Subject and the Object entities in the *query sentence*. The Subject and the Object entities are indicated with the subject and object tags, respectively. Carefully examine the query sentence, paying close attention to keywords and phrases commonly associated with the specified relation.

Consider the following:

*   **Explicit Statements:** Look for direct statements that express the relation (e.g., "is a subsidiary for," "was born in").
*   **Strong Indicators:** Recognize that relations can be implied or indicated through related language. For example, "native of" often implies "city of birth," and "subsidiary for" directly indicates an organizational relationship.
*   **Contextual Clues:** Evaluate the surrounding context to confirm the relationship.

Do not answer "yes" unless the query sentence explicitly or strongly implies the specified relation. Base your decision solely on the content of the query sentence. Ignore any support examples provided.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_qwen_rpo_node_y_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

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

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_qwen_rpo_node_y_gradpo-gen_cross --prompt 'You are given a relation name, a description of the relation in brackets, a example exemplifying the relation, and a query sentence.

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
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_rpo_node_x_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support instance that exemplifies the relation, and a query sentence. The relation connects a **subject entity** and an **object entity**, which are marked with the <subject> and <object> tags, respectively. Your task is to determine whether the relation described in the support instance holds between the subject and object entities in the query sentence. 

**Key Steps:**
1. **Identify the subject and object entities** in the query sentence (they are marked with <subject> and <object> tags).
2. **Analyze the relation** based on the provided description and the support instance. The relation must be a direct, specific connection between the subject and object entities as defined in the description.
3. **Check the query sentence** for explicit evidence of the relation. If the subject and object entities in the query are not connected by the relation described (e.g., the object is not the correct entity type, or the connection is not explicitly stated), answer "no."

**Example:**  
For the relation "director," the subject (e.g., a film) must be connected to the object (e.g., a person) through the relation described in the support instance. If the object in the query is a genre or a different entity type, the answer is "no."

If the relation holds between the subject and object entities in the query sentence, answer "yes"; otherwise, answer "no."'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_rpo_node_x_gradpo-gen_cross --prompt 'You are told a relation name, a description of the relation in quotes, a support instance that exemplifies the relation, and a query text. The relation connects a **subject entity** and an **object entity**, which are marked with the <subject> and <object> tags, respectively. Your task is to determine whether the relation described in the support instance holds between the subject and object entities in the query text. 

**Key Steps:**
1. **Identify the subject and object entities** in the query text (they are marked with <subject> and <object> tags).
2. **Analyze the relation** based on the provided description and the support instance. The relation must be a direct, specific connection between the subject and object entities as defined in the description.
3. **Check the query text** for explicit evidence of the relation. If the subject and object entities in the query are not connected by the relation described (e.g., the object is not the correct entity type, or the connection is not explicitly stated), answer "no."

**Example:**  
For the relation "director," the subject (e.g., a film) must be connected to the object (e.g., a person) through the relation described in the support instance. If the object in the query is a genre or a different entity type, the answer is "no."

If the relation holds between the subject and object entities in the query text, answer "yes"; otherwise, answer "no."'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_rpo_node_y_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support instance illustrating the relation, and a query sentence. The relation connects a **subject** entity (marked with <subject> tags) and an **object** entity (marked with <object> tags). Your task is to determine whether the query sentence explicitly expresses the relation defined by the relation name and description.

**Key Guidelines:**
1. **Understand the relation'"'"'s definition precisely**: 
   - For "original network", the subject (e.g., a program or show) must be explicitly tied to the object (e.g., a TV network or platform) as its original distribution channel. This includes direct mentions of the network or platform in the sentence.
   - For "residence", the subject (e.g., a person or entity) must be explicitly tied to the object (e.g., a location) as a place they live or are associated with.
   - For "instance of", the subject must be explicitly described as a specific example of the object (e.g., a song being part of an album, a dam being a type of structure). This requires a direct part-whole or category relationship, not a location-based or functional one.

2. **Focus on the subject and object tags**: Identify the subject and object entities in the query sentence as per the tags. Ensure you do not confuse the subject with a role or the object with a location unless explicitly defined by the relation. For example, "BBC Three" in the "original network" relation is a network entity, not a location.

3. **Avoid conflating association with location or function**: 
   - "Original network" requires a direct link between the subject (e.g., a show) and the object (e.g., a network) as its original platform.
   - "Residence" requires a location-based connection (e.g., a person living in a city).
   - "Instance of" requires a part-whole or category relationship (e.g., a song being part of an album, a dam being a type of structure), not a location or functional relationship.

4. **Check for explicitness**: The relation must be explicitly stated or implied by the sentence'"'"'s structure. For example, "Schatz was inaugurated as Hawaii'"'"'s lieutenant governor" explicitly links the role (subject) to the jurisdiction (object), satisfying "applies to jurisdiction." Similarly, "The Smoking Room" (subject) is explicitly tied to "BBC Three" (object) as a sitcom aired on that network, satisfying "original network."

**Example Clarification**: 
- In the "original network" relation, the object must be a network or platform (e.g., "BBC Three", "Canal 13") that directly distributes the subject'"'"'s work. 
- In the "instance of" relation, the subject must be a specific example of the object (e.g., a song on an album, a dam in a country), not a location or category.

Answer "yes" if the query sentence explicitly expresses the relation as defined; otherwise, answer "no."'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_rpo_node_y_gradpo-gen_cross --prompt 'You are given a relation label, a description of the relation in curly braces, a support occurrence illustrating the relation, and a query sentence. The relation connects a **subject** entity (marked with <subject> tags) and an **object** entity (marked with <object> tags). Your task is to determine whether the query sentence clearly indicates the relation defined by the relation label and description.

**Key Guidelines:**
1. **Understand the relation'"'"'s definition precisely**: 
   - For "original network", the subject (e.g., a program or show) must be explicitly tied to the object (e.g., a TV network or platform) as its original distribution channel. This includes direct mentions of the network or platform in the sentence.
   - For "residence", the subject (e.g., a person or entity) must be explicitly tied to the object (e.g., a location) as a place they live or are associated with.
   - For "instance of", the subject must be explicitly described as a specific example of the object (e.g., a song being part of an album, a dam being a type of structure). This requires a direct part-whole or category relationship, not a location-based or functional one.

2. **Focus on the subject and object tags**: Identify the subject and object entities in the query sentence as per the tags. Ensure you do not confuse the subject with a role or the object with a location unless explicitly defined by the relation. For example, "BBC Three" in the "original network" relation is a network entity, not a location.

3. **Avoid conflating association with location or function**: 
   - "Original network" requires a direct link between the subject (e.g., a show) and the object (e.g., a network) as its original platform.
   - "Residence" requires a location-based connection (e.g., a person living in a city).
   - "Instance of" requires a part-whole or category relationship (e.g., a song being part of an album, a dam being a type of structure), not a location or functional relationship.

4. **Check for explicitness**: The relation must be explicitly stated or implied by the sentence'"'"'s structure. For example, "Schatz was inaugurated as Hawaii'"'"'s lieutenant governor" explicitly links the role (subject) to the jurisdiction (object), satisfying "applies to jurisdiction." Similarly, "The Smoking Room" (subject) is explicitly tied to "BBC Three" (object) as a sitcom aired on that network, satisfying "original network."

**Example Clarification**: 
- In the "original network" relation, the object must be a network or platform (e.g., "BBC Three", "Canal 13") that directly distributes the subject'"'"'s work. 
- In the "instance of" relation, the subject must be a specific example of the object (e.g., a song on an album, a dam in a country), not a location or category.

Answer "yes" if the query sentence clearly indicates the relation as defined; otherwise, answer "no."'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_rpo_node_x_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. These are entities – consider their types and roles within the sentence. The prompt provides a description of the relation – use this in conjunction with the support instance to guide your reasoning.

You need to decide whether the relation holds between the Subject and the Object entities in the query sentence. The relation may be explicitly stated or implied through the context of the sentence. Carefully consider the surrounding words and phrases to determine if a connection exists.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_rpo_node_x_gradpo-gen_cross --prompt 'You are given a concept, a description of the relation in brackets, a example sentence exemplifying the relation, and a question.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. These are entities – consider their types and roles within the sentence. The prompt provides a description of the relation – use this in conjunction with the example sentence to guide your reasoning.

You need to decide whether the relation holds between the Subject and the Object entities in the question. The relation may be explicitly stated or implied through the context of the sentence. Carefully consider the surrounding words and phrases to determine if a connection exists.

If the relation holds between the Subject and the Object entities in the question, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_rpo_node_y_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively.
You need to determine if the query sentence expresses the given relation between the Subject and the Object entities, even if it'"'"'s not stated directly.

Consider the support instance as an example of the relation. However, it might not be directly applicable to every query. Focus on understanding the *type* of relationship described in the support instance, not on finding a word-for-word match.

When evaluating the query sentence, focus on the core connection between the Subject and the Object. Ignore irrelevant details or specific locations unless they are essential to establishing the relation. Ask yourself: Does this sentence, at its heart, describe the same kind of relationship as the support instance?

If the query sentence expresses the relation between the Subject and the Object entities, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_rpo_node_y_gradpo-gen_cross --prompt 'You are given a relation name, a description of the relation in brackets, a sentence serving as an example of the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively.
You need to determine if the query sentence expresses certain relation between the Subject and the Object entities, even if it'"'"'s not stated directly.

Consider the instance as an example of the relation. However, it might not be directly applicable to every query. Focus on understanding the *type* of relationship described in the support instance, not on finding a word-for-word match.

When evaluating the query sentence, focus on the core connection between the Subject and the Object. Ignore irrelevant details or specific locations unless they are essential to establishing the relation. Ask yourself: Does this sentence, at its heart, describe the same kind of relationship as the support instance?

If the query sentence expresses the relation between the Subject and the Object entities, answer "yes"; otherwise, answer "no".'
