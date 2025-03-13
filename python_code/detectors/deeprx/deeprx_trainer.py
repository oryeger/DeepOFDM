from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deeprx.deeprx_detector import DeepRxDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, N_USERS, N_ANTS, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
import numpy as np
from python_code.utils.constants import MOD_PILOT, SHOW_ALL_ITERATIONS, EPOCHS, NUM_BITS, ITERATIONS, N_ANTS

import commpy.modulation as mod

Softmax = torch.nn.Softmax(dim=1)

class DeepRxTrainer(Trainer):

    def __init__(self, num_res):
        self.lr = 5e-3
        super().__init__(num_res)

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self, num_res):
        self.detector = [DeepRxDetector(18,2,1).to(DEVICE) for _ in
                         range(N_USERS)]  # 2D list for Storing the DeepSIC Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor) -> list[float]:
        """
        Trains a DeepSIC Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        for _ in range(EPOCHS):
            soft_estimation, llrs = single_model(rx)
            if MOD_PILOT <= 2:
                tx_reshaped = tx
            else:
                tx_reshaped = tx.reshape(int(tx.shape[0] // NUM_BITS), NUM_BITS,tx.shape[1])

            train_samples = int(soft_estimation.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop(soft_estimation[:train_samples], tx_reshaped[:train_samples])
            val_loss = self._calculate_loss(soft_estimation[train_samples:], tx_reshaped[train_samples:])
            val_loss = val_loss.item()
            loss += current_loss
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect

    def _train_models(self, model: List[List[DeepRxDetector]], i: int, tx_all: List[torch.Tensor],
                      rx_all: List[torch.Tensor]):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(N_USERS):
            train_loss_vect , val_loss_vect = self._train_model(model[user][i], tx_all[user], rx_all[user].to(DEVICE))
            if user == 0:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor):
        """
        Main training function for DeepSIC trainer.
        """

        tx_all, rx_all = self._prepare_data_for_training(tx, rx_real)
        # Training the DeepSIC network for each user for iteration=1
        train_loss_vect , val_loss_vect = self._train_models(self.detector, 0, tx_all, rx_all)
        # Initializing the probabilities
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec, llrs_mat = self._calculate_posteriors(self.detector, i, rx_real) # This is after the weights and biases have been updated
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self._prepare_data_for_training(tx, rx_real.to('cuda'))
            # Training the DeepSIC networks for the iteration>1
            train_loss_cur , val_loss_cur =  self._train_models(self.detector, i, tx_all, rx_all)
            if SHOW_ALL_ITERATIONS:
                train_loss_vect = train_loss_vect + train_loss_cur
                val_loss_vect = val_loss_vect + val_loss_cur
        return train_loss_vect , val_loss_vect

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        est_rs = est.squeeze(-1)
        return self.criterion(input=est_rs, target=tx)

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, rx: torch.Tensor) -> torch.Tensor:
        # detect and decode
        for i in range(ITERATIONS):
            rx_in = rx.to('cuda').unsqueeze(-1)
            probs_vec, llrs = self._calculate_posteriors(self.detector, i + 1, rx_in)

        return self._compute_output(probs_vec), llrs



    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for user in range(N_USERS):
            rx_all.append(rx.unsqueeze(-1))
            tx_all.append(tx[:, user, :])
        return tx_all, rx_all

    def _calculate_posteriors(self, model: List[List[nn.Module]], i: int, rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(rx.shape[0],rx.shape[1]-2*N_ANTS,rx.shape[2],rx.shape[3]).to(DEVICE)
        llrs_mat = torch.zeros(next_probs_vec.shape).to(DEVICE)
        for user in range(N_USERS):
            with torch.no_grad():
                output, llrs = model[user][i - 1](rx)
            index_start = user*NUM_BITS
            index_end = (user+1) * NUM_BITS
            next_probs_vec[:, index_start:index_end,:,:] = output
            llrs_mat[:, index_start:index_end,:,:] = llrs
        return next_probs_vec, llrs_mat

