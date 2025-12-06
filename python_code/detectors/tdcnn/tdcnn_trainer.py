from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.tdcnn.tdcnn_detector import TDCNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable
from torch.optim import Adam
from torch.nn import MSELoss
Softmax = torch.nn.Softmax(dim=1)

class TDCNNTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        self.lr = 5e-3
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'TDCNN'

    def _initialize_detector(self, num_bits, n_users, n_ants):

        self.detector = TDCNNDetector(num_bits, n_users).to(DEVICE)

    def _train_model(self, single_model: nn.Module, s_t_matrix_clean: torch.Tensor, s_t_matrix: torch.Tensor, epochs: int) -> list[float]:
        """
        Trains a TDCNN Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)

        self.optimizer = Adam(filter(lambda p: p.requires_grad, single_model.parameters()), lr=self.lr)
        self.criterion = MSELoss().to(DEVICE)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        # for _ in range(epochs):
        for epoch in range(epochs):
            s_t_matrix_estimation = single_model(s_t_matrix)
            train_samples = int(s_t_matrix.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop_tdcnn(s_t_matrix_estimation[:train_samples], s_t_matrix_clean[:train_samples])
            val_loss = self._calculate_loss(s_t_matrix_estimation[train_samples:], s_t_matrix_clean[train_samples:])
            val_loss = val_loss.item()
            loss += current_loss
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect




    def _online_training(self, s_t_matrix_clean: torch.Tensor, s_t_matrix: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor, stage: str = "base"):
        """
        Main training function for TDCNN trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """


        # Training the TDCNN network for each user for iteration=1
        train_loss_vect , val_loss_vect = self._train_model(self.detector, s_t_matrix_clean, s_t_matrix, epochs)
        return train_loss_vect , val_loss_vect

    def run_train_loop_tdcnn(self, s_t_matrix: torch.Tensor, s_t_matrix_clean: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        loss = self._calculate_loss(est=s_t_matrix, tx=s_t_matrix_clean)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return current_loss


    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        est_rs = est.squeeze(-1)
        return self.criterion(input=est_rs, target=tx)

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, s_t_matrix: torch.Tensor, num_bits: int, n_users: int, iterations: int, probs_in: torch.Tensor) -> tuple[List, List]:
        # detect and decode
        s_t_matrix_proc = self.detector(s_t_matrix)
        return s_t_matrix_proc, torch.tensor([], device=DEVICE)

