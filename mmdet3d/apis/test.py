import mmcv
import torch
import numpy as np


def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        #print("results : ", result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    
    return results