import json

PROMPT_TEMPLATE = dict()

PROMPT_TEMPLATE['1_shots_one_prompt'] = '''Relation name: “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#

Query sentence: #QUERY_SENTENCE#

Please decide the relation between the subject and the object in the query sentence holds or not with the relation given at the top. An example of the relation is given. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags. Only output "yes" or "no" depending on the that relation in the query sentence holds or not.'''

PROMPT_TEMPLATE['1_shots_wo_rule_ws'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”

2. Relation “#RELATION_2#”

3. Relation “#RELATION_3#”

4. Relation “#RELATION_4#”

5. Relation “#RELATION_5#”

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_wo_rule_ws_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

The description for each of the five relations are given. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_short_reason'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations. The examples are given is this format "[Object] ... [Subject]". A relation always connects a subject and object entity.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_short_reason_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples and relation description for each of the five relations. The examples are given is this format "[Object] ... [Subject]". A relation always connects a subject and object entity.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_wo_rule'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_w_rule'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples with rules for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_w_rule_syn_expl'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, a rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_w_rule_w_des_syn_expl'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, a rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_k_gen_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
#GEN_SUPPORT_SENTENCE_1#
Sentence 5: #SUPPORT_SENTENCE_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
#GEN_SUPPORT_SENTENCE_2#
Sentence 5: #SUPPORT_SENTENCE_2#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
#GEN_SUPPORT_SENTENCE_3#
Sentence 5: #SUPPORT_SENTENCE_3#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
#GEN_SUPPORT_SENTENCE_4#
Sentence 5: #SUPPORT_SENTENCE_4#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
#GEN_SUPPORT_SENTENCE_5#
Sentence 5: #SUPPORT_SENTENCE_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_k_gen_wo_rule'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
#GEN_SUPPORT_SENTENCE_1#
Sentence 5: #SUPPORT_SENTENCE_1#

2. Relation “#RELATION_2#”
#GEN_SUPPORT_SENTENCE_2#
Sentence 5: #SUPPORT_SENTENCE_2#

3. Relation “#RELATION_3#”
#GEN_SUPPORT_SENTENCE_3#
Sentence 5: #SUPPORT_SENTENCE_3#

4. Relation “#RELATION_4#”
#GEN_SUPPORT_SENTENCE_4#
Sentence 5: #SUPPORT_SENTENCE_4#

5. Relation “#RELATION_5#”
#GEN_SUPPORT_SENTENCE_5#
Sentence 5: #SUPPORT_SENTENCE_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_cot_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Output the reason followed by only the name of the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_cot2_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Output your solution in the following format:
Relation in query sentence: The object "[OBJECT from the sentence]" ..(try to find a neutral relation (don't try to relate here from the given 5 type of relations). if no realation, just say there is no relation between them).. the subject "[SUBJECT from the sentence]" 
Relation 1: "#RELATION_1#" does / does not match, since it connects #RELATION_DESCRIPTION_1#. (keep the explanation short and the matching should be exact to the detail)
Relation 2: "#RELATION_2#" does / does not match, since it connects #RELATION_DESCRIPTION_2#. (keep the explanation short and the matching should be exact to the detail)
Relation 3: "#RELATION_3#" does / does not match, since it connects #RELATION_DESCRIPTION_3#. (keep the explanation short and the matching should be exact to the detail)
Relation 4: "#RELATION_4#" does / does not match, since it connects #RELATION_DESCRIPTION_4#. (keep the explanation short and the matching should be exact to the detail)
Relation 5: "#RELATION_5#" does / does not match, since it connects #RELATION_DESCRIPTION_5#. (keep the explanation short and the matching should be exact to the detail)
Answer: [Write the name of the relation that matched above] (if no matches, then ouput "no_relation")'''

PROMPT_TEMPLATE['1_shots_cot3_wo_rule_w_des'] = ''' You are given an example of few-shot relation extraction problem and how to solve it below:

Question: Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: per:parents, org:founded_by, per:date_of_death, org:members, per:origin. If there is no relation between the subject and the object of the query sentence or none of the five relations are correct, choose no_relation.

Note that, your chosen relation should not be the closest or assume anything but a perfect and exact match to the query with its description.

Here are some examples (support sentences) for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “per:parents”
Relation Description: a person's parents
Support sentence 1: <subject>Prince William</subject> is the son of <object>Prince Charles</object> and <object>Princess Diana</object>.

2. Relation “org:founded_by”
Relation Description: the person who founded an organization
Support sentence 1: <subject>Microsoft</subject> was founded by <object>Bill Gates</object> and Paul Allen in 1975.

3. Relation “per:date_of_death”
Relation Description: a person's date of death
Support sentence 1: <subject>Albert Einstein</subject> died on <object>April 18, 1955</object> in Princeton, New Jersey.

4. Relation “org:members”
Relation Description: members belonging to an organization
Support sentence 1: <subject>The European Union</subject> includes member countries like <object>Germany</object>, <object>France</object>, and <object>Italy</object>.

5. Relation “per:origin”
Relation Description: the nationality or ethnic origin of a person
Support sentence 1: <subject>Elon Musk</subject> is originally from <object>South Africa</object>.

Query sentence: <subject>Marie Curie</subject> was born in Poland but later moved to France, where she died in <object>1934</object>.

Solution: 

<query_connection>
    <subject>Marie Curie</subject>
    <subject entity>Person</subject entity>
    <object>1934</object>
    <object entity>Date</object entity>
    <is subject object related>yes</is subject object related>
    <subject object connection>object is the date when the subject had died.</subject object connection>
</query_connection>

<relation_explanation>
    <Relation 1>  
        <subject entity requirement>Person</subject entity requirement>
        <object entity requirement>Person</object entity requirement>
        <subject object connection requirement>object is the parent of the subject</subject object connection requirement>
        <match with query>this relation is about object being the parent of the subject but the query's object is the date of death of the subject</match with query>
        <match decision>no</match decision>
    </Relation 1>
    <Relation 2>  
        <subject entity requirement>Person</subject entity requirement>
        <object entity requirement>Organization</object entity requirement>
        <subject object connection requirement>object is the founder of the subject</subject object connection requirement>
        <match with query>this relation is about object being the founder of the subject but the query's object is the date of death of the subject</match with query>
        <match decision>no</match decision>
    </Relation 2>
    <Relation 3>  
        <subject entity requirement>Person</subject entity requirement>
        <object entity requirement>Date</object entity requirement>
        <subject object connection requirement>object is the date of death of the subject</subject object connection requirement>
        <match with query>this relation is about the object being the date of death of the subject and the query's object is also the date of death of the subject. the query's subject and object's entity requirements is also matched</match with query>
        <match decision>yes</match decision>
    </Relation 3>
    <Relation 4>  
        <subject entity requirement>Organization</subject entity requirement>
        <object entity requirement>Organization</object entity requirement>
        <subject object connection requirement>object is a member of the subject</subject object connection requirement>
        <match with query>this relation is about object being a member of the subject but the query's object is the date of death of the subject</match with query>
        <match decision>no</match decision>
    </Relation 4>
    <Relation 5>  
        <subject entity requirement>Person</subject entity requirement>
        <object entity requirement>Location</object entity requirement>
        <subject object connection requirement>object can be a nationality or ethnic origin of the subject</subject object connection requirement>
        <match with query>this relation is about object being a nationality or ethnic origin of the person but the query's object is the date of death of the subject</match with query>
        <match decision>no</match decision>
    </Relation 5>
</relation_explanation>

<answer>per:date_of_death</answer>

In the exact format of the above solution, solve the given problem below and don't output anything more:

Question: Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If there is no relation between the subject and the object of the query sentence or none of the five relations are correct, choose no_relation.

Note that, your chosen relation should not be the closest or assume anything but a perfect and exact match to the query with its description.

Here are some examples (support sentences) for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

'''


PROMPT_TEMPLATE['1_shots_wo_rule_w_des_w_reason'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. Moreover, a description on relation between the subject and the object is also given for each sentence. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Description on relation between subject and object of support sentence 1: #REASON_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Description on relation between subject and object of support sentence 1: #REASON_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Description on relation between subject and object of support sentence 1: #REASON_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Description on relation between subject and object of support sentence 1: #REASON_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Description on relation between subject and object of support sentence 1: #REASON_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_w_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples with rules for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_w_rule_vb_syn_expl_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, the verbalization of the rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['1_shots_w_rule_vb_syn_expl'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, the verbalization of the rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_1_1#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_2_1#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_3_1#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_4_1#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_5_1#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_wo_rule'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_w_rule'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples with rules for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Rule for sentence 2: #RULE_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Rule for sentence 3: #RULE_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Rule for sentence 4: #RULE_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Rule for sentence 5: #RULE_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Rule for sentence 2: #RULE_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Rule for sentence 3: #RULE_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Rule for sentence 4: #RULE_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Rule for sentence 5: #RULE_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Rule for sentence 2: #RULE_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Rule for sentence 3: #RULE_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Rule for sentence 4: #RULE_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Rule for sentence 5: #RULE_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Rule for sentence 2: #RULE_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Rule for sentence 3: #RULE_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Rule for sentence 4: #RULE_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Rule for sentence 5: #RULE_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Rule for sentence 2: #RULE_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Rule for sentence 3: #RULE_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Rule for sentence 4: #RULE_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Rule for sentence 5: #RULE_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_w_rule_syn_expl'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples with rules for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, a rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Rule for sentence 2: #RULE_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Rule for sentence 3: #RULE_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Rule for sentence 4: #RULE_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Rule for sentence 5: #RULE_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Rule for sentence 2: #RULE_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Rule for sentence 3: #RULE_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Rule for sentence 4: #RULE_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Rule for sentence 5: #RULE_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Rule for sentence 2: #RULE_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Rule for sentence 3: #RULE_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Rule for sentence 4: #RULE_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Rule for sentence 5: #RULE_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Rule for sentence 2: #RULE_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Rule for sentence 3: #RULE_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Rule for sentence 4: #RULE_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Rule for sentence 5: #RULE_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Rule for sentence 2: #RULE_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Rule for sentence 3: #RULE_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Rule for sentence 4: #RULE_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Rule for sentence 5: #RULE_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_w_rule_w_des_syn_expl'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, a rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Rule for sentence 2: #RULE_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Rule for sentence 3: #RULE_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Rule for sentence 4: #RULE_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Rule for sentence 5: #RULE_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Rule for sentence 2: #RULE_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Rule for sentence 3: #RULE_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Rule for sentence 4: #RULE_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Rule for sentence 5: #RULE_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Rule for sentence 2: #RULE_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Rule for sentence 3: #RULE_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Rule for sentence 4: #RULE_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Rule for sentence 5: #RULE_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Rule for sentence 2: #RULE_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Rule for sentence 3: #RULE_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Rule for sentence 4: #RULE_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Rule for sentence 5: #RULE_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Rule for sentence 2: #RULE_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Rule for sentence 3: #RULE_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Rule for sentence 4: #RULE_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Rule for sentence 5: #RULE_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['10_shots_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Support sentence 6: #SUPPORT_SENTENCE_1_6#
Support sentence 7: #SUPPORT_SENTENCE_1_7#
Support sentence 8: #SUPPORT_SENTENCE_1_8#
Support sentence 9: #SUPPORT_SENTENCE_1_9#
Support sentence 10: #SUPPORT_SENTENCE_1_10#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Support sentence 6: #SUPPORT_SENTENCE_2_6#
Support sentence 7: #SUPPORT_SENTENCE_2_7#
Support sentence 8: #SUPPORT_SENTENCE_2_8#
Support sentence 9: #SUPPORT_SENTENCE_2_9#
Support sentence 10: #SUPPORT_SENTENCE_2_10#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Support sentence 6: #SUPPORT_SENTENCE_3_6#
Support sentence 7: #SUPPORT_SENTENCE_3_7#
Support sentence 8: #SUPPORT_SENTENCE_3_8#
Support sentence 9: #SUPPORT_SENTENCE_3_9#
Support sentence 10: #SUPPORT_SENTENCE_3_10#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Support sentence 6: #SUPPORT_SENTENCE_4_6#
Support sentence 7: #SUPPORT_SENTENCE_4_7#
Support sentence 8: #SUPPORT_SENTENCE_4_8#
Support sentence 9: #SUPPORT_SENTENCE_4_9#
Support sentence 10: #SUPPORT_SENTENCE_4_10#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Support sentence 6: #SUPPORT_SENTENCE_5_6#
Support sentence 7: #SUPPORT_SENTENCE_5_7#
Support sentence 8: #SUPPORT_SENTENCE_5_8#
Support sentence 9: #SUPPORT_SENTENCE_5_9#
Support sentence 10: #SUPPORT_SENTENCE_5_10#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['9_shots_wo_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Support sentence 6: #SUPPORT_SENTENCE_1_6#
Support sentence 7: #SUPPORT_SENTENCE_1_7#
Support sentence 8: #SUPPORT_SENTENCE_1_8#
Support sentence 9: #SUPPORT_SENTENCE_1_9#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Support sentence 6: #SUPPORT_SENTENCE_2_6#
Support sentence 7: #SUPPORT_SENTENCE_2_7#
Support sentence 8: #SUPPORT_SENTENCE_2_8#
Support sentence 9: #SUPPORT_SENTENCE_2_9#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Support sentence 6: #SUPPORT_SENTENCE_3_6#
Support sentence 7: #SUPPORT_SENTENCE_3_7#
Support sentence 8: #SUPPORT_SENTENCE_3_8#
Support sentence 9: #SUPPORT_SENTENCE_3_9#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Support sentence 6: #SUPPORT_SENTENCE_4_6#
Support sentence 7: #SUPPORT_SENTENCE_4_7#
Support sentence 8: #SUPPORT_SENTENCE_4_8#
Support sentence 9: #SUPPORT_SENTENCE_4_9#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Support sentence 6: #SUPPORT_SENTENCE_5_6#
Support sentence 7: #SUPPORT_SENTENCE_5_7#
Support sentence 8: #SUPPORT_SENTENCE_5_8#
Support sentence 9: #SUPPORT_SENTENCE_5_9#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_w_rule_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples with rules for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Rule for sentence 1: #RULE_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Rule for sentence 2: #RULE_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Rule for sentence 3: #RULE_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Rule for sentence 4: #RULE_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Rule for sentence 5: #RULE_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Rule for sentence 1: #RULE_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Rule for sentence 2: #RULE_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Rule for sentence 3: #RULE_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Rule for sentence 4: #RULE_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Rule for sentence 5: #RULE_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Rule for sentence 1: #RULE_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Rule for sentence 2: #RULE_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Rule for sentence 3: #RULE_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Rule for sentence 4: #RULE_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Rule for sentence 5: #RULE_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Rule for sentence 1: #RULE_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Rule for sentence 2: #RULE_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Rule for sentence 3: #RULE_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Rule for sentence 4: #RULE_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Rule for sentence 5: #RULE_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Rule for sentence 1: #RULE_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Rule for sentence 2: #RULE_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Rule for sentence 3: #RULE_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Rule for sentence 4: #RULE_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Rule for sentence 5: #RULE_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_w_rule_vb_syn_expl'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, the verbalization of the rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

PROMPT_TEMPLATE['5_shots_w_rule_vb_syn_expl_w_des'] = '''Please choose the correct relation between the entities in the query sentence below. You can choose between five relations: #RELATION_LIST#. If none of the five relations are correct, choose no_relation.

Here are some examples (support sentences) for each of the five relations along with the description of the relation. A relation always connects a subject and object entity. We indicate the entities with <subject> and <object> tags.

Additionally, the verbalization of the rule for each support sentence is provided. A rule is the sequence of token constraints for the words that lies between the shortest path in the dependency tree between the subject and the object. Each token contraint is surrounded by third brackets and is composed of a token type with a value, where token type can be a lemma, word, pos tag, etc., or a token constraint can be empty meaning that is can be any word. Each constraint can also be followed by a Regex quantifier.

1. Relation “#RELATION_1#”
Relation Description: #RELATION_DESCRIPTION_1#
Support sentence 1: #SUPPORT_SENTENCE_1_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_1_1#
Support sentence 2: #SUPPORT_SENTENCE_1_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_1_2#
Support sentence 3: #SUPPORT_SENTENCE_1_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_1_3#
Support sentence 4: #SUPPORT_SENTENCE_1_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_1_4#
Support sentence 5: #SUPPORT_SENTENCE_1_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_1_5#

2. Relation “#RELATION_2#”
Relation Description: #RELATION_DESCRIPTION_2#
Support sentence 1: #SUPPORT_SENTENCE_2_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_2_1#
Support sentence 2: #SUPPORT_SENTENCE_2_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_2_2#
Support sentence 3: #SUPPORT_SENTENCE_2_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_2_3#
Support sentence 4: #SUPPORT_SENTENCE_2_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_2_4#
Support sentence 5: #SUPPORT_SENTENCE_2_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_2_5#

3. Relation “#RELATION_3#”
Relation Description: #RELATION_DESCRIPTION_3#
Support sentence 1: #SUPPORT_SENTENCE_3_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_3_1#
Support sentence 2: #SUPPORT_SENTENCE_3_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_3_2#
Support sentence 3: #SUPPORT_SENTENCE_3_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_3_3#
Support sentence 4: #SUPPORT_SENTENCE_3_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_3_4#
Support sentence 5: #SUPPORT_SENTENCE_3_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_3_5#

4. Relation “#RELATION_4#”
Relation Description: #RELATION_DESCRIPTION_4#
Support sentence 1: #SUPPORT_SENTENCE_4_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_4_1#
Support sentence 2: #SUPPORT_SENTENCE_4_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_4_2#
Support sentence 3: #SUPPORT_SENTENCE_4_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_4_3#
Support sentence 4: #SUPPORT_SENTENCE_4_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_4_4#
Support sentence 5: #SUPPORT_SENTENCE_4_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_4_5#

5. Relation “#RELATION_5#”
Relation Description: #RELATION_DESCRIPTION_5#
Support sentence 1: #SUPPORT_SENTENCE_5_1#
Verbalized rule for sentence 1: #RULE_SENTENCE_5_1#
Support sentence 2: #SUPPORT_SENTENCE_5_2#
Verbalized rule for sentence 2: #RULE_SENTENCE_5_2#
Support sentence 3: #SUPPORT_SENTENCE_5_3#
Verbalized rule for sentence 3: #RULE_SENTENCE_5_3#
Support sentence 4: #SUPPORT_SENTENCE_5_4#
Verbalized rule for sentence 4: #RULE_SENTENCE_5_4#
Support sentence 5: #SUPPORT_SENTENCE_5_5#
Verbalized rule for sentence 5: #RULE_SENTENCE_5_5#

Query sentence: #QUERY_SENTENCE#

Only output the correct relation, nothing else.'''

RELATION_DESCRIPTION = {
    "org:alternate_names": "an organization's alternate names",
    "org:city_of_headquarters": "an organization's city of headquarters",
    "org:country_of_headquarters": "an organization's country of headquarters",
    "org:dissolved": "an organization's date of dissolution",
    "org:founded": "an organization's date of founding",
    "org:founded_by": "an organization's founder",
    "org:member_of": "an organization's membership of another entity",
    "org:members": "an organization's members",
    "org:number_of_employees/members": "an organization's number of employees or members",
    "org:parents": "an organization's parents",
    "org:political/religious_affiliation": "an organization's political or religious affiliation",
    "org:shareholders": "an organization's shareholders",
    "org:stateorprovince_of_headquarters": "an organization's state or province of headquarters",
    "org:subsidiaries": "an organization's subsidiaries",
    "org:top_members/employees": "an organization's top members or employees",
    "org:website": "an organization's website",
    "per:age": "a person's age",
    "per:alternate_names": "a person's alternate names",
    "per:cause_of_death": "a person's cause of death",
    "per:charges": "a person's criminal charges",
    "per:children": "a person's children",
    "per:cities_of_residence": "a person's cities of residence",
    "per:city_of_birth": "a person's city of birth",
    "per:city_of_death": "a person's city of death",
    "per:countries_of_residence": "a person's countries of residence",
    "per:country_of_birth": "a person's country of birth",
    "per:country_of_death": "a person's country of death",
    "per:date_of_birth": "a person's date of birth",
    "per:date_of_death": "a person's date of death",
    "per:employee_of": "a person's employer",
    "per:origin": "a person's city or country of origin",
    "per:other_family": "a person's other family",
    "per:parents": "a person's parents",
    "per:religion": "a person's religion",
    "per:schools_attended": "schools attended by a person",
    "per:siblings": "a person's siblings",
    "per:spouse": "a person's spouse",
    "per:stateorprovince_of_birth": "a person's state or province of birth",
    "per:stateorprovince_of_death": "a person's state or province of death",
    "per:stateorprovinces_of_residence": "a person's state or province of residence",
    "per:title": "a person's title",

    ### RE-tacred
    # "org:city_of_branch": "an organization's city of branch",
    # "org:stateorprovince_of_branch": "an organization's state or province of branch",
    # "org:country_of_branch": "an organization's country of branch",
    # "per:identity": "a person's identity",
}

RELATION_DESCRIPTION_FSTACRED = {
    "org:alternate_names": "Alternate names of the organization including former names, aliases, alternate spellings, acronyms, abbreviations, translations or transliterations of names, and any official designators such as stock ticker code or airline call sign.",
    "org:city_of_headquarters": "Location of the headquarters of the organization at the city, town, or village level.",
    "org:country_of_headquarters": "Countries in which the headquarters of the organization are located.",
    "org:dissolved": "The date on which the organization was dissolved.",
    "org:founded": "The date on which the organization was founded.",
    "org:founded_by": "The person, organization, or geopolitical entity that founded the organization.",
    "org:member_of": "Organizations or geopolitical entities of which the organization is a member itself.",
    "org:members": "Organizations or Geopolitical entities that are members of the organization.",
    "org:number_of_employees/members": "The total number of people who are employed by or have membership in an organization.",
    "org:parents": "Organizations or geopolitical entities of which the organization is a subsidiary.",
    "org:political/religious_affiliation": "Ideological groups with which the organization is associated.",
    "org:shareholders": "Any organization, person, or geopolitical entity that holds shares (majority or not) of the organization.",
    "org:stateorprovince_of_headquarters": "Location of the headquarters of the organization at the state or province level.",
    "org:subsidiaries": "Organizations that are subsidiaries of the organization.",
    "org:top_members/employees": "The persons in high-level, leading positions at the organization.",
    "org:website": "An official top level URL for the organization's website.",
    "per:age": "A reported age of the person.",
    "per:alternate_names": "Names used to refer to the person that are distinct from the official name. Alternate names may include aliases, stage names, alternate transliterations, abbreviations, alternate spellings, nicknames, or birth names.",
    "per:cause_of_death": "The explicit cause of death for the person.",
    "per:charges": " The charges or crimes (alleged or convicted) of the person.",
    "per:children": "The children of the person, including adopted and step-children.",
    "per:cities_of_residence": "Geopolitical entities at the level of city, town, or village in which the person has lived.",
    "per:city_of_birth": " The geopolitical entity at the municipality level (city, town, or village) in which the person was born.",
    "per:city_of_death": "The geopolitical entity at the level of city, town, village in which the person died.",
    "per:countries_of_residence": "All countries in which the person has lived.",
    "per:country_of_birth": "The country in which the person was born.",
    "per:country_of_death": "The country in which the person died.",
    "per:date_of_birth": "The date on which the person was born.",
    "per:date_of_death": "The date of the person's death.",
    "per:employee_of": "The organizations or geopolitical entities (governments) by which the person has been employed.",
    "per:origin": "The nationality and/or ethnicity of the person.",
    "per:other_family": "Other family members of the person including brothers-in-law, sisters-in-law, grandparents, grandchildren, cousins, aunts, uncles, etc.",
    "per:parents": "The parents of the person.",
    "per:religion": "The religion to which the person has belonged.",
    "per:schools_attended": "Any school (college, high school, university, etc.) that the person has attended.",
    "per:siblings": "The brothers and sisters of the person.",
    "per:spouse": "The spouse or spouses of the person.",
    "per:stateorprovince_of_birth": "The geopolitical entity at state or province level in which the person was born.",
    "per:stateorprovince_of_death": "The geopolitical entity at state or province level in which the person died.",
    "per:stateorprovinces_of_residence": "Geopolitical entities at the state or province level in which the person has lived.",
    "per:title": " Official or unofficial name(s) of the employment or membership positions that have been held by the person.",
}

RELATION_DESCRIPTION_NYT29 = {
    "/location/administrative_division/country": "The country of an administrative division.",
    "/location/country/capital": "The capital city of a country.",
    "/location/country/administrative_divisions": "The administrative divisions within a country.",
    "/location/neighborhood/neighborhood_of": "The larger location a neighborhood is part of.",
    "/location/location/contains": "One location contains another.",
    "/people/person/nationality": "The nationality of a person.",
    "/people/person/place_lived": "Places where a person has lived.",
    "/people/deceased_person/place_of_death": "The place where a person died.",
    "/business/person/company": "The company a person is linked to.",
    "/location/us_state/capital": "The capital city of a US state.",
    "/people/person/place_of_birth": "The birthplace of a person.",
    "/people/person/children": "The children of a person.",
    "/business/company/founders": "The founders of a company.",
    "/business/company/place_founded": "The founding location of a company.",
    "/sports/sports_team/location": "The location or home of a sports team.",
    "/people/person/ethnicity": "The ethnicity of a person.",
    "/people/ethnicity/geographic_distribution": "Where people of a specific ethnicity are located.",
    "/people/person/religion": "The religion of a person.",
    "/business/company/major_shareholders": "The major shareholders of a company.",
    "/location/province/capital": "The capital of a province.",
    "/location/br_state/capital": "The capital of a Brazilian state.",
    "/business/company/advisors": "The advisors of a company.",
    "/film/film_location/featured_in_films": "Locations featured in films.",
    "/film/film/featured_film_locations": "Films that have showcased specific locations.",
    "/location/us_county/county_seat": "The administrative center of a US county.",
    "/time/event/locations": "The locations of events.",
    "/people/deceased_person/place_of_burial": "The burial place of a person.",
    "/people/place_of_interment/interred_here": "People interred at a specific location.",
    "/business/company_advisor/companies_advised": "The companies advised by a company advisor."
}

RELATION_ENTITIES = {
    "org:alternate_names": ("organization", "organization"),
    "org:city_of_headquarters": ("organization", "city"),
    "org:country_of_headquarters": ("organization", "country"),
    "org:dissolved": ("organization", "date"),
    "org:founded": ("organization", "date"),
    "org:founded_by": ("organization", "person"),
    "org:member_of": ("organization", "organization"),
    "org:members": ("organization", "person or organization"),
    "org:number_of_employees/members": ("organization", "number"),
    "org:parents": ("organization", "organization"),
    "org:political/religious_affiliation": ("organization", "group or organization"),
    "org:shareholders": ("organization", "person or organization"),
    "org:stateorprovince_of_headquarters": ("organization", "state or province"),
    "org:subsidiaries": ("organization", "organization"),
    "org:top_members/employees": ("organization", "person or organization"),
    "org:website": ("organization", "website"),
    "per:age": ("person", "number"),
    "per:alternate_names": ("person", "person"),
    "per:cause_of_death": ("person", "cause_of_death"),
    "per:charges": ("person", "criminal_charge"),
    "per:children": ("person", "person"),
    "per:cities_of_residence": ("person", "city"),
    "per:city_of_birth": ("person", "city"),
    "per:city_of_death": ("person", "city"),
    "per:countries_of_residence": ("person", "country"),
    "per:country_of_birth": ("person", "country"),
    "per:country_of_death": ("person", "country"),
    "per:date_of_birth": ("person", "date"),
    "per:date_of_death": ("person", "date"),
    "per:employee_of": ("person", "organization"),
    "per:origin": ("person", "location"),
    "per:other_family": ("person", "person"),
    "per:parents": ("person", "person"),
    "per:religion": ("person", "religion"),
    "per:schools_attended": ("person", "organization"),
    "per:siblings": ("person", "person"),
    "per:spouse": ("person", "person"),
    "per:stateorprovince_of_birth": ("person", "state or province"),
    "per:stateorprovince_of_death": ("person", "state or province"),
    "per:stateorprovinces_of_residence": ("person", "state or province"),
    "per:title": ("person", "title")
}

RELATION_ENTITIES_NYT29 = {
    "/location/administrative_division/country": ("location", "country"),
    "/location/country/capital": ("country", "location"),
    "/location/country/administrative_divisions": ("country", "location"),
    "/location/neighborhood/neighborhood_of": ("location", "location"),
    "/location/location/contains": ("location", "location"),
    "/people/person/nationality": ("person", "location or misc"),
    "/people/person/place_lived": ("person", "location"),
    "/people/deceased_person/place_of_death": ("person", "location"),
    "/business/person/company": ("person", "organization"),
    "/location/us_state/capital": ("US state", "location"),
    "/people/person/place_of_birth": ("person", "location"),
    "/people/person/children": ("person", "person"),
    "/business/company/founders": ("organization", "person"),
    "/business/company/place_founded": ("organization", "location"),
    "/sports/sports_team/location": ("organization", "location"),
    "/people/person/ethnicity": ("person", "misc"),
    "/people/ethnicity/geographic_distribution": ("misc", "location"),
    "/people/person/religion": ("person", "misc"),
    "/business/company/major_shareholders": ("organization", "person"),
    "/location/province/capital": ("province", "location"),
    "/location/br_state/capital": ("location", "location"),
    "/business/company/advisors": ("organization", "person"),
    "/film/film_location/featured_in_films": ("location", "misc"),
    "/film/film/featured_film_locations": ("misc", "location"),
    "/location/us_county/county_seat": ("US county", "location"),
    "/time/event/locations": ("misc", "location"),
    "/people/deceased_person/place_of_burial": ("person", "location"),
    "/people/place_of_interment/interred_here": None,
    "/business/company_advisor/companies_advised": ("person", "organization")
}

RELATION_DESCRIPTION_FEWREL_ORIGINAL = {
    "sport": "sport in which the subject participates or belongs to",
    "crosses": "obstacle (body of water, road, ...) which this bridge crosses over or this tunnel goes under",
    "military rank": "military rank achieved by a person (should usually have a \"start time\" qualifier), or military rank associated with a position",
    "mother": "female parent of the subject. For stepmother, use \"stepparent\" (P3448)",
    "child": "subject has object as biological, foster, and/or adoptive child",
    "constellation": "the area of the celestial sphere of which the subject is a part (from a scientific standpoint, not an astrological one)",
    "follows": "immediately prior item in a series of which the subject is a part [if the subject has replaced the preceding item, e.g. political offices, use \"replaces\" (P1365)]",
    "member of": "organization or club to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a position such as a member of parliament (use P39 for that).",
    "original language of film or TV show": "language in which a film or a performance work was originally created. Deprecated for written works; use P407 (\"language of work or name\") instead.",
    "spouse": "the subject has the object as their spouse (husband, wife, partner, etc.). Use \"partner\" (P451) for non-married companions",
    "competition class": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "position played on team / speciality": "position or specialism of a player on a team, e.g. Small Forward",
    "located in or next to body of water": "sea, lake or river",
    "voice type": "person's voice type. expected values: soprano, mezzo-soprano, contralto, countertenor, tenor, baritone, bass (and derivatives)",
    "main subject": "primary topic of a work (see also P180: depicts)",
    "part of": "object of which the subject is a part (it's not useful to link objects which are themselves parts of other objects already listed as parts of the subject). Inverse property of \"has part\" (P527, see also \"has parts of the class\" (P2670)).",
}

RELATION_DESCRIPTION_FEWREL_PROCESSED = {
    "sport": "sport in which the subject participates or belongs to",
    "crosses": "obstacle (body of water, road, ...) which this bridge crosses over or this tunnel goes under",
    "military rank": "military rank achieved by a person (should usually have a \"start time\" qualifier), or military rank associated with a position",
    "mother": "female parent of the subject.",
    "child": "subject has object as biological, foster, and/or adoptive child",
    "constellation": "the area of the celestial sphere of which the subject is a part (from a scientific standpoint, not an astrological one)",
    "follows": "immediately prior item in a series of which the subject is a part",
    "member of": "organization or club to which the subject belongs.",
    "original language of film or TV show": "language in which a film or a performance work was originally created.",
    "spouse": "the subject has the object as their spouse (husband, wife, partner, etc.).",
    "competition class": "official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion",
    "position played on team / speciality": "position or specialism of a player on a team, e.g. Small Forward",
    "located in or next to body of water": "sea, lake or river",
    "voice type": "person's voice type.",
    "main subject": "primary topic of a work",
    "part of": "object of which the subject is a part (it's not useful to link objects which are themselves parts of other objects already listed as parts of the subject).",
}

RELATION_ENTITIES_FEWREL = {
    "sport": ("person", "sport"),
    "crosses": ("structure", "body of water or road"),
    "military rank": ("person", "military rank"),
    "mother": ("person", "person"),
    "child": ("person", "person"),
    "constellation": ("astronomical object", "constellation"),
    "follows": ("entity", "entity"),
    "member of": ("person", "organization"),
    "original language of film or TV show": ("creative work", "language"),
    "spouse": ("person", "person"),
    "competition class": ("sporting entity", "competition class"),
    "position played on team / speciality": ("athlete", "position"),
    "located in or next to body of water": ("geospatial entity", "body of water"),
    "voice type": ("person", "voice type"),
    "main subject": ("creative work", "subject"),
    "part of": ("entity", "entity"),
}

# with open('mayank_anchor_rules.json', 'r') as f:
#     RULE_MAP = json.load(f)

# with open('sentence_map.json', 'r') as f:
#     SENTENCE_MAP = json.load(f)

# with open('reason_map.json', 'r') as f:
#     REASON_MAP = json.load(f)

# with open('short_reason_map.json', 'r') as f:
#     SHORT_REASON_MAP = json.load(f)

# with open('query_short_reason_map.json', 'r') as f:
#     QUERY_SHORT_REASON_MAP = json.load(f)

# with open('generated_examples_using_rules.json', 'r') as f:
#     K_GEN_SEN_MAP = json.load(f)

# def get_prompt_template(prompt_name):
#     assert prompt_name in PROMPT_TEMPLATE.keys()
    
#     return PROMPT_TEMPLATE[prompt_name]

def get_relation_description(relation, dt = "tacred"):
    if dt == "tacred":
        assert relation in RELATION_DESCRIPTION.keys()

        return RELATION_DESCRIPTION[relation]
    elif dt == "nyt29":
        assert relation in RELATION_DESCRIPTION_NYT29.keys()

        return RELATION_DESCRIPTION_NYT29[relation]
    elif dt == "fewrel":
        assert relation in RELATION_DESCRIPTION_FEWREL_PROCESSED.keys()

        return RELATION_DESCRIPTION_FEWREL_PROCESSED[relation]

# def get_rule_for_sentence(idx):
#     assert idx in RULE_MAP

#     return RULE_MAP[idx]

# def get_reason_for_sentence(idx):
#     assert idx in REASON_MAP

#     return REASON_MAP[idx]

# def get_sentences():
#     return SENTENCE_MAP

# def get_k_gen_sentences(idx):
#     assert idx in K_GEN_SEN_MAP

#     return K_GEN_SEN_MAP[idx]

# def get_relation_entities(idx, dt = "tacred"):
#     if dt == "tacred":
#         assert idx in RELATION_ENTITIES.keys()

#         return RELATION_ENTITIES[idx]
#     elif dt == "nyt29":
#         assert idx in RELATION_ENTITIES_NYT29.keys()

#         return RELATION_ENTITIES_NYT29[idx]
#     elif dt == "fewrel":
#         assert idx in RELATION_ENTITIES_FEWREL.keys()

#         return RELATION_ENTITIES_FEWREL[idx]

