import yaml
import json

# Load YAML from cfg.yaml
with open('video_emo.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# Convert YAML to JSON
json_data = json.dumps(yaml_data, indent=2)

# Save JSON to file
with open('cfg.json', 'w') as json_file:
    json_file.write(json_data)
