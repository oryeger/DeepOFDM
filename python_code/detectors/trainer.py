import random
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import BCELoss
from torch.nn import BCEWithLogitsLoss

from torch.optim import Adam

from python_code import DEVICE, conf
from python_code.utils.probs_utils import relevant_indices


random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


torch.autograd.set_detect_anomaly(True)


class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self, num_bits, n_users, n_ants):
        # initialize matrices, dataset and detector
        self.lr = conf.learning_rate
        self.is_online_training = True
        #  self._initialize_dataloader(num_res,self.pilot_size)
        self._initialize_detector(num_bits, n_users, n_ants)

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self, num_bits, n_users, n_ants):
        """
        Every trainer must have some base detector
        """
        self.detector = None

    # calculate train loss
    def _calculate_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def _deep_learning_setup(self,single_model):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = Adam(filter(lambda p: p.requires_grad, single_model.parameters()), lr=self.lr)
        self.criterion = BCEWithLogitsLoss().to(DEVICE)


    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor, n_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor) -> Tuple[List[float], List[float]]:
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iterations: int, probs_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass


    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor, first_half_flag) -> float:
        # calculate loss
        if first_half_flag:
            est_cur = est[:,0::2,:,:]
            tx_cur = tx[:,0::2,:]
        else:
            est_cur = est
            tx_cur = tx

        # if conf.make_64QAM_16QAM:
        #     indices = relevant_indices(tx_cur.shape[1], 1.5)
        # else:
        #     indices = list(range(0, tx_cur.shape[1]))
        indices = list(range(0, tx_cur.shape[1]))

        loss = self._calculate_loss(est=est_cur[:,indices,:,:], tx=tx_cur[:,indices,:])
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss