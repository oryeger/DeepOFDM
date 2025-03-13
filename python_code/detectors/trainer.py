import random
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from python_code import DEVICE, conf
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.constants import NUM_BITS, NUM_SYMB_PER_SLOT




random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

def get_next_divisible(num, divisor):
    return (num + divisor - 1) // divisor * divisor

class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self, num_res):
        # initialize matrices, dataset and detector
        self.lr = 5e-3
        self.is_online_training = True
        pilot_size = get_next_divisible(conf.pilot_size,NUM_BITS*NUM_SYMB_PER_SLOT)
        self.pilot_size = pilot_size
        self._initialize_dataloader(num_res,self.pilot_size)
        self._initialize_detector(num_res)

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self, num_res):
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
        self.criterion = BCELoss().to(DEVICE)


    def _initialize_dataloader(self, num_res, pilot_size):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        conf.num_res = num_res
        self.channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                   pilots_length=pilot_size,
                                                   blocks_num=conf.blocks_num,
                                                   num_res=conf.num_res,
                                                   fading_in_channel=conf.fading_in_channel,
                                                   spatial_in_channel=conf.spatial_in_channel,
                                                   delayspread_in_channel=conf.delayspread_in_channel,
                                                   clip_percentage_in_tx=conf.clip_percentage_in_tx,
                                                   cfo=conf.cfo,
                                                   go_to_td=conf.go_to_td,
                                                   cfo_in_rx=conf.cfo_in_rx,
                                                   kernel_size=conf.kernel_size)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def _forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass


    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self._calculate_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss
