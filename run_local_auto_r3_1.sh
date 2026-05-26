python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:2 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_rpo_node_x_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support instance that exemplifies the relation, and a query sentence. The relation connects a **subject entity** and an **object entity**, which are marked with the <subject> and <object> tags, respectively. Your task is to determine whether the relation described in the support instance holds between the subject and object entities in the query sentence. 

**Key Steps:**
1. **Identify the subject and object entities** in the query sentence (they are marked with <subject> and <object> tags).
2. **Analyze the relation** based on the provided description and the support instance. The relation must be a direct, specific connection between the subject and object entities as defined in the description.
3. **Check the query sentence** for explicit evidence of the relation. If the subject and object entities in the query are not connected by the relation described (e.g., the object is not the correct entity type, or the connection is not explicitly stated), answer "no."

**Example:**  
For the relation "director," the subject (e.g., a film) must be connected to the object (e.g., a person) through the relation described in the support instance. If the object in the query is a genre or a different entity type, the answer is "no."

If the relation holds between the subject and object entities in the query sentence, answer "yes"; otherwise, answer "no."'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:2 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_rpo_node_y_gradpo-gen_cross --prompt 'You are given a relation label, a description of the relation in curly braces, a support occurrence illustrating the relation, and a query sentence. The relation connects a **subject** entity (marked with <subject> tags) and an **object** entity (marked with <object> tags). Your task is to determine whether the query sentence clearly indicates the relation defined by the relation label and description.

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
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:2 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_rpo_node_x_cross --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. These are entities – consider their types and roles within the sentence. The prompt provides a description of the relation – use this in conjunction with the support instance to guide your reasoning.

You need to decide whether the relation holds between the Subject and the Object entities in the query sentence. The relation may be explicitly stated or implied through the context of the sentence. Carefully consider the surrounding words and phrases to determine if a connection exists.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:2 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_rpo_node_x_gradpo-gen_cross --prompt 'You are given a concept, a description of the relation in brackets, a example sentence exemplifying the relation, and a question.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. These are entities – consider their types and roles within the sentence. The prompt provides a description of the relation – use this in conjunction with the example sentence to guide your reasoning.

You need to decide whether the relation holds between the Subject and the Object entities in the question. The relation may be explicitly stated or implied through the context of the sentence. Carefully consider the surrounding words and phrases to determine if a connection exists.

If the relation holds between the Subject and the Object entities in the question, answer "yes"; otherwise, answer "no".'
