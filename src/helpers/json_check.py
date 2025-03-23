import json
from pprint import pprint

# Path to your JSON file
json_path = 'dataset/TextOCR/TextOCR_0.1_train.json'

# Load and print the structure
with open(json_path, 'r') as f:
    data = json.load(f)

# Print the top-level keys
print("Top-level keys in JSON:")
pprint(list(data.keys()))

# Example: print first image annotation
print("\nSample entry:")
if 'anns' in data:
    first_ann_id = list(data['anns'].keys())[0]
    pprint(data['anns'][first_ann_id])
elif 'images' in data:
    first_image_id = list(data['images'].keys())[0]
    pprint(data['images'][first_image_id])
else:
    pprint(data)
