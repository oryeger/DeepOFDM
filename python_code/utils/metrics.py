import torch
import numpy as np
from python_code.utils.constants import NUM_BITS



def calculate_ber(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    index_list = [[] for _ in range(NUM_BITS)]

    if (prediction.size() != target.size()):
        prediction_rs = torch.empty_like(target)
        for i in range(0,NUM_BITS):
            index_list[i] = list(range(i, prediction.shape[1], NUM_BITS))

        for j in range(prediction.shape[0]):
            for i in range(0, NUM_BITS):
                prediction_rs[j*NUM_BITS+i,:] = prediction[j,index_list[i]]
        prediction   = prediction_rs

    ber = 1 - torch.mean(torch.all(prediction == target, dim=1).float()).item()
    return ber
