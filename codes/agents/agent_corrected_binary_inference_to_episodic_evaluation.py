import json, sys
from pathlib import Path
sys.path.append('codes')
from agents.agent_scorer import score, NO_RELATION
from agents.agent_data_loader import load_split_episodes

json_path = Path('gradients_experiments/gradient_baseline_train_node12_top3_k5.json')
obj = json.load(open(json_path, 'r', encoding='utf-8'))

variant = next(v for v in obj['prompt_region_editing']['generated_prompt_variants'] if v.get('revised_prompt'))
idx = variant['generation_index']
preds = variant['full_evaluation']['predicted_labels']

dataset_type = obj['dataset_type']
query_index = obj['args']['query_index']
full_eval_split = obj['train_gradient_collection']['evaluation_split']
data_dir = obj['args']['data_dir']

dataset = load_split_episodes(split=full_eval_split, data_dir=data_dir, dataset_type=dataset_type)
episodes = dataset['episodes']
shots = dataset['shots']
queries = dataset['queries']

episode_relations = []
episode_labels = []
for episode in episodes:
    ways = episode['meta_train']
    query_id = episode['meta_test'][query_index]
    query = queries[query_id]
    query_relation = query['relation']
    relations = [shots[way[0]]['relation'] for way in ways]
    episode_relations.append(relations)
    episode_labels.append(query_relation if query_relation in relations else NO_RELATION)

episode_predictions = []
offset = 0
for relations in episode_relations:
    chunk = preds[offset: offset + len(relations)]
    offset += len(relations)
    predicted_relation = NO_RELATION
    for relation, pred in zip(relations, chunk):
        if pred == 'yes':
            predicted_relation = relation
            break
    episode_predictions.append(predicted_relation)

p, r, f1 = score(episode_labels, episode_predictions, verbose=False)
print({
    'generation_index': idx,
    'precision': p * 100.0,
    'recall': r * 100.0,
    'f1': f1 * 100.0,
})
