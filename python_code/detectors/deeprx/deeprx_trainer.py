from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deeprx.deeprx_detector import DeepRxDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import TRAIN_PERCENTAGE, NUM_SYMB_PER_SLOT, N_ANTS
from python_code.utils.probs_utils import prob_to_BPSK_symbol

Softmax = torch.nn.Softmax(dim=1)

class DeepRxTrainer(Trainer):

    def __init__(self, num_res: int, n_users: int):
        self.lr = 5e-3
        super().__init__(num_res, n_users)

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self, num_bits, n_users):
        self.detector = [DeepRxDetector(18,2,1, 64, num_bits, N_ANTS ).to(DEVICE) for _ in
                         range(n_users)]  # 2D list for Storing the DeepSIC Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor, num_bits: int, epochs: int) -> list[float]:
        """
        Trains a DeepSIC Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        train_loss_vect = []
        val_loss_vect = []
        for _ in range(epochs):
            soft_estimation, llrs = single_model(rx)
            tx_reshaped = tx.reshape(int(tx.shape[0] // num_bits), num_bits,tx.shape[1])
            train_samples = int(soft_estimation.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop(soft_estimation[:train_samples], tx_reshaped[:train_samples])
            val_loss = self._calculate_loss(soft_estimation[train_samples:], tx_reshaped[train_samples:])
            val_loss = val_loss.item()
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect

    def _train_models(self, model: List[DeepRxDetector], tx_all: List[torch.Tensor],
                      rx_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(n_users):
            train_loss_vect , val_loss_vect = self._train_model(model[user], tx_all[user], rx_all[user].to(DEVICE), num_bits,epochs)
            if user == 0:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int):
        """
        Main training function for DeepSIC trainer.
        """

        tx_all, rx_all = self._prepare_data_for_training(tx, rx_real, n_users)
        train_loss_vect , val_loss_vect = self._train_models(self.detector, tx_all, rx_all, num_bits, n_users, epochs)
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

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iterations: int) -> torch.Tensor:
        # detect and decode
        rx_in = rx.to('cuda').unsqueeze(-1)
        probs_vec, llrs = self._calculate_posteriors(self.detector, rx_in, num_bits, n_users)

        return self._compute_output(probs_vec), llrs



    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, n_users: int) -> [
        torch.Tensor, torch.Tensor]:

        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        rx_split = torch.split(rx, NUM_SYMB_PER_SLOT, dim=0)
        rx_stacked = torch.stack(rx_split, dim=0)
        rx_permuted = rx_stacked.permute(0, 2, 3, 1)
        for user in range(n_users):
            # rx_all.append(rx_permuted)
            rx_all.append(rx.unsqueeze(-1))
            cur_tx = tx[:, user, :]
            tx_split = torch.split(cur_tx, NUM_SYMB_PER_SLOT, dim=0)
            tx_stacked = torch.stack(tx_split, dim=0)
            tx_permuted = tx_stacked.permute(0, 2, 1)
            # tx_all.append(tx_permuted)
            tx_all.append(cur_tx)
        return tx_all, rx_all

    def _calculate_posteriors(self, model: List[List[List[nn.Module]]], rx: torch.Tensor, num_bits: int, n_users: int) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(rx.shape[0],num_bits*n_users,rx.shape[2],rx.shape[3]).to(DEVICE)
        llrs_mat = torch.zeros(next_probs_vec.shape).to(DEVICE)
        for user in range(n_users):
            with torch.no_grad():
                output, llrs = model[user](rx)
            index_start = user*num_bits
            index_end = (user+1) * num_bits
            next_probs_vec[:, index_start:index_end,:,:] = output
            llrs_mat[:, index_start:index_end,:,:] = llrs
        return next_probs_vec, llrs_mat

