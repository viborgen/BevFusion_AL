import orjson as json 
import os



train_path = '/cvrr/BevFusion_AL/data/nuscenes/v1.0-trainval'
test_path = '/cvrr/BevFusion_AL/data/nuscenes/v1.0-test'
u_path = '/cvrr/BevFusion_AL/data/nuscenes/v1.0-unlabeled'

# Load the JSON files into Python dictionaries
with open(os.path.join(train_path, 'scene.json'), 'r') as f:
    trainval_scenes = json.loads(f.read())
with open(os.path.join(test_path, 'scene.json'), 'r') as f:
    test_scenes = json.loads(f.read())
with open(os.path.join(u_path, 'scene.json'), 'r') as f:
    unlabeled_scenes = json.loads(f.read())

# Extract the scene names
trainval_names = {scene['name'] for scene in trainval_scenes}
test_names = {scene['name'] for scene in test_scenes}
unlabeled_names = {scene['name'] for scene in unlabeled_scenes}

# Find scene names that occur in more than one file
common_names = trainval_names.intersection(test_names, unlabeled_names)

print("Common scene names: ", common_names)