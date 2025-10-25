import json

data_path = "./results/history/nodes_3/merge_83d/merged_observations_83d.json"
expert_space_path = "./configs/config_space/expert_space.json"

with open(expert_space_path, 'r') as f:
    expert_space = json.load(f)
expert_spark_dims = set(expert_space.get('spark', []))

with open(data_path, 'r') as f:
    data = json.load(f)

observations = data.get('observations', [])
print(len(observations))

better_configs_with_objective = []
threshold = 8600
for observation in observations:
    config = observation.get('config', {})
    objective = observation.get('objectives', [None])[0] if observation.get('objectives') else None
    if objective < threshold:
        filtered_config = {key: value for key, value in config.items() if key in expert_spark_dims}
        better_configs_with_objective.append((filtered_config, objective))

print(f"Total better configs: {len(better_configs_with_objective)}")

better_configs_with_objective.sort(key=lambda x: x[1])

configs_part1 = []
configs_part2 = []
total_obj_part1 = 0
total_obj_part2 = 0

for config, objective in better_configs_with_objective:
    if total_obj_part1 <= total_obj_part2:
        configs_part1.append(config)
        total_obj_part1 += objective
    else:
        configs_part2.append(config)
        total_obj_part2 += objective

print(f"Part 1: {len(configs_part1)} configs, total objective: {total_obj_part1:.2f}")
print(f"Part 2: {len(configs_part2)} configs, total objective: {total_obj_part2:.2f}")

with open('./results/history/nodes_3/merge_83d/better_configs_83d_part1.json', 'w') as f:
    json.dump(configs_part1, f, indent=2, ensure_ascii=False)

with open('./results/history/nodes_3/merge_83d/better_configs_83d_part2.json', 'w') as f:
    json.dump(configs_part2, f, indent=2, ensure_ascii=False)