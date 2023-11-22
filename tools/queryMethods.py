from typing import Dict, List
import orjson as json
import random
import os

import subprocess
import sys

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import init_detector, inference_detector

from mmdet3d.apis import single_gpu_test
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataloader, build_dataset

import torch
from mmcv import Config
from mmdet.apis import init_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset


from nuscenes.eval.prediction.data_classes import Prediction
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
from mmdet.apis import multi_gpu_test, set_random_seed
from torchpack.utils.config import configs
import argparse
from mmdet.datasets import replace_ImageToTensor
from torchpack import distributed as dist


def randomQ(random_num, random_labels):
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
#____________________________________________________________________________________________________
    # Calculate probabilities and entropy for each result
    for result in results:
        for class_name, class_output in list(result.items()):
            # The scores are the last column in the class_output array
            scores = result['scores_3d'].numpy()

            # Convert the scores to probabilities
            probabilities = torch.softmax(scores, dim=0)

            # Calculate the entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities)).item()

            # Add the entropy to the result
            result[class_name + '_entropy'] = entropy

    # Sort the results by entropy
    results.sort(key=lambda x: x[class_name + '_entropy'], reverse=True)

    selected_samples = results[:AL_split]
    print("entropy selected samples: ", selected_samples)

    # Select the samples with the highest entropy
    return selected_samples




def inference(cfg_file):
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

    #dist.init()

    # torch.backends.cudnn.benchmark = True
    # torch.cuda.set_device(dist.local_rank())

    latest = run_latest_model()

    # Initialize the detector
    #model = init_detector(cfg, f'./checkpoints/{latest}/latest.pth', device='cuda:0')

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

    return results