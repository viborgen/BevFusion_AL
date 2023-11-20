# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List

from nuscenes import NuScenes

import orjson



files = ['train', 'val', 'test', 'unlabeled']

# List to store name tokens
train = []
val = []
unlabeled = []
test = []

for file in files:
    # Open the JSON file
    with open(f'/cvrr/bevfusion_bjork/tools/json_map/{file}.json', 'r') as json_file:
        data = orjson.loads(json_file.read())

    #Search for name tokens in the JSON data
        for item in data:
            if 'name' in item:
                if file == 'train':
                    train.append(item['name'])
                elif file == 'val':
                    val.append(item['name'])
                elif file == 'unlabeled':
                    unlabeled.append(item['name'])
                else:
                    test.append(item['name'])
                

    # Print the list of name tokens
print("\n List of name tokens in the train file:", train)
print("\n List of name tokens in the val file:", val)
print("\n List of name tokens in the test file:", test)
print(f'Len train: {len(train)}')
print(f'Len val: {len(val)}')
print(f'Len test: {len(test)}')
print(f'Len unlabeled: {len(unlabeled)}')


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    # if split in {'train', 'val', 'train_detect', 'train_track'}:
    #     assert version.endswith('trainval'), \
    #         'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    # elif split in {'mini_train', 'mini_val'}:
    #     assert version.endswith('mini'), \
    #         'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    # elif split == 'test':
    #     assert version.endswith('test'), \
    #         'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    # else:
    #     raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    -Labeled(Train)/Unlabeled/val/test: (700(split between labeled/unlabeled changes per round)/ 150/ 150 scenes)
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test + unlabeled
    assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test, 'unlabeled': unlabeled}#,
                    #'mini_train': mini_train, 'mini_val': mini_val,
                    #'train_detect': train_detect, 'train_track': train_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
