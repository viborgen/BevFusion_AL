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

def entropy(probs):
    return -torch.sum(probs * torch.log(probs), dim=-1)

def active_learning_query(model, unlabeled_loader):
    model.eval()
    entropy_scores = []
    dataset = unlabeled_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    #with torch.no_grad():
    for data in unlabeled_loader:
        with torch.no_grad():
            # Get the model's output probabilities
            output_probs = torch.nn.functional.softmax(model(return_loss=False, rescale=True, **data), dim=-1)

            # Calculate the entropy of the output probabilities
            batch_entropy_scores = entropy(output_probs)

        entropy_scores.extend(batch_entropy_scores)

        batch_size = len(batch_entropy_scores)
        for _ in range(batch_size):
            prog_bar.update()

    # Get the indices of the data points with the highest entropy(decending order)
    query_indices = np.argsort(entropy_scores.cpu().numpy())[::-1]
    print(query_indices)

    return query_indices