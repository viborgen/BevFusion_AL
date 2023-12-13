from typing import Dict, List
import orjson as json
import random
import os
import numpy as np

import subprocess
import sys

import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import init_detector, inference_detector

#from mmdet3d.apis import single_gpu_test
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

import torch
from mmcv import Config
from mmdet.apis import init_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset


from nuscenes.eval.prediction.data_classes import Prediction
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet.apis import multi_gpu_test, set_random_seed
from torchpack.utils.config import configs
import argparse
from mmdet.datasets import replace_ImageToTensor
from torchpack import distributed as dist

u_path = '/cvrr/BevFusion_AL/data/nuscenes/v1.0-unlabeled'
trainval_path = '/cvrr/BevFusion_AL/data/nuscenes/v1.0-trainval'
jsonfile_prefix = './tools/'

with open(os.path.join(u_path, 'sample.json'), 'r') as f:
        tokens = json.loads(f.read())


with open(os.path.join(u_path, 'scene.json'), 'r') as f:
        scene_tokens = json.loads(f.read())


def config_setup(cfg_file):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("-config", metavar="FILE", default=f"{cfg_file}", help="config file")
    args, opts = parser.parse_known_args()

    # if "LOCAL_RANK" not in os.environ:
    #     os.environ["MASTER_HOST"] = str(args.local_rank)

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    return cfg, args



def randomQ(random_num, random_labels, seed=None):
    if seed is not None:
        random.seed(seed)
    train = random.sample(list(random_labels), random_num)

    return train



def run_latest_model():
    # Get a list of all directories in the directory
    dirs = [d for d in os.listdir('./checkpoints') if os.path.isdir(os.path.join('./checkpoints', d))]

    # Sort the directories by modification time
    dirs.sort(key=lambda x: os.path.getmtime(os.path.join('./checkpoints', x)))

    # Get the latest directory
    latest_dir = dirs[-1]

    return latest_dir


def entropyQ(AL_split, cfg_file):
    
    #print("results: ", results)
    results = inference(cfg_file)
    cfg, args = config_setup(cfg_file)
     # build the dataloader
    samples_per_gpu = 1
    dataset = build_dataset(cfg.data.unlabeled)
    #print("dataset: ", dataset)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    result_files, _ = dataset.format_results(results, jsonfile_prefix)

    print('loading json file')
    with open(os.path.join(jsonfile_prefix, 'results_nusc.json'), 'r') as f:
        results = json.loads(f.read())

    # print('len results: ', len(results)) 
    # print('len keys: ', len(results.keys()))
    # print('len values: ', len(results.values()))
    # print('len items: ', len(results.items()))   


    sample_tokens = []
    detection_scores = []

    keys = list(results.keys())
    values = list(results.values())
    items = list(results.items())
    key = keys[1]
    value = values[1]
    #item = items[0]
    print('key name : ', key)
    #print('items name : ', item)
    print('value name : ', len(value.items()))


    for item in values[1].values():
        #print('value name : ', item[1]['detection_score'])
        sample_tokens.append(item[1]['sample_token'])
        detection_scores.append(item[1]['detection_score'])


    #print('sample tokens: ', sample_tokens)
    #print('detection scores: ', detection_scores)

    print('is loaded')
    entropys = []

    for detection_score in detection_scores:
       
        # Calculate the entropy of the 'detection_scores'
        probabilities = torch.tensor(detection_score)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()
        entropys.append(entropy)

        #print(f'Entropy of detection_scores: {entropy}')
    #print('entropys: ', entropys)

    # Initialize an empty list
    results_list = []

    # Iterate over the sample tokens, detection scores, and entropys
    for sample_token, detection_score, entropy in zip(sample_tokens, detection_scores, entropys):
        # Add a new dictionary to the list for each sample token
        results_list.append({
            'token': {
                'sample_token': sample_token,
                'detection_score': detection_score,
                'entropy': entropy
            }
        })

    # Sort the results_list by entropy in descending order
    sorted_results_list = sorted(results_list, key=lambda x: x['token']['entropy'], reverse=True)

    #print('Sorted results list: ', sorted_results_list)
    #____________________________________________________________________________________________________

    scenes = []

    scene_names = []

    for result in sorted_results_list:
        sample = result['token']['sample_token']

        for token in tokens:
            if  token.get('token') == sample and token.get('scene_token') not in scenes:
                scenes.append(token.get('scene_token'))

            # Break the loop if we have found 100 unique scenes
            if len(scenes) >= AL_split:
                break

        # Break the outer loop as well
        if len(scenes) >= AL_split:
            break
        
    for scene in scenes:
        for scene_token in scene_tokens:
            if scene_token.get('token') == scene:
                scene_names.append(scene_token.get('name'))
                break
    #print("scene name: ", scene_names) 



    #print("scene names: ", scenes)
    return scene_names


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    #print("results: ", results)
    return results



def inference(cfg_file):
    cfg, args = config_setup(cfg_file)

    latest = run_latest_model()

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.unlabeled, dict):
        cfg.data.unlabeled.test_mode = True
        samples_per_gpu = cfg.data.unlabeled.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.unlabeled.pipeline = replace_ImageToTensor(cfg.data.unlabeled.pipeline)
    elif isinstance(cfg.data.unlabeled, list):
        for ds_cfg in cfg.data.unlabeled:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.unlabeled]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.unlabeled:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.unlabeled)
    #print("dataset: ", dataset)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, f'./checkpoints/{latest}/latest.pth', map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        results = single_gpu_test(model, data_loader)
    
    # results = dataset._extract_data(results, cfg.data.unlabeled.pipeline, "interval")
    # print(results)

    return results

