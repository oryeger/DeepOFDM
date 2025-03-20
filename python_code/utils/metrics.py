import torch


def calculate_ber(prediction: torch.Tensor, target: torch.Tensor, num_bits: int) -> float:
    """
    Returns the calculated ber of the prediction and the target (ground truth transmitted word)
    """
    prediction = prediction.long()
    target = target.long()
    index_list = [[] for _ in range(num_bits)]

    if (prediction.size() != target.size()):
        prediction_rs = torch.empty_like(target)
        for i in range(0,num_bits):
            index_list[i] = list(range(i, prediction.shape[1], num_bits))

        for j in range(prediction.shape[0]):
            for i in range(0, num_bits):
                prediction_rs[j*num_bits+i,:] = prediction[j,index_list[i]]
        prediction   = prediction_rs

    ber = 1 - torch.mean(torch.all(prediction == target, dim=1).float()).item()
    return ber
