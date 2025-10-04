from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deepsicmb.deepsicmb_detector import DeepSICMBDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable
import numpy as np

Softmax = torch.nn.Softmax(dim=1)

class DeepSICMBTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        self.lr = 5e-3
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'DeepSICNB'

    def _initialize_detector(self, num_bits, n_users, n_ants):
        self.detector = [[[DeepSICMBDetector(num_bits, n_users).to(DEVICE) for _ in range(conf.iterations )] for _ in range(n_users)] for _ in
                         range(conf.num_res)]  # 2D list for Storing the DeepSICMB Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor, num_bits:int, epochs: int) -> list[float]:
        """
        Trains a VSDNN Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        y_total = self._preprocess(rx)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        for _ in range(epochs):
            soft_estimation, llrs = single_model(y_total)
            if conf.mod_pilot <= 2:
                tx_reshaped = tx
            else:
                tx_reshaped = tx.reshape(int(tx.numel() // num_bits), num_bits)

            train_samples = int(soft_estimation.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop(soft_estimation[:train_samples], tx_reshaped[:train_samples], False)
            val_loss = self._calculate_loss(soft_estimation[train_samples:], tx_reshaped[train_samples:])
            val_loss = val_loss.item()
            loss += current_loss
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect

    def _train_models(self, model: List[List[List[DeepSICMBDetector]]], i: int, tx_all: List[torch.Tensor],
                      rx_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for re in range(conf.num_res):
            for user in range(n_users):
                train_loss_vect , val_loss_vect = self._train_model(model[re][user][i], tx_all[user][:,re], rx_all[user][:,:,re].to(DEVICE), num_bits, epochs)
                if user == 0:
                    train_loss_vect_user = train_loss_vect
                    val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor):
        """
        Main training function for DeepSICMB trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        initial_probs = self._initialize_probs(tx, num_bits, n_users)
        tx_all, rx_all = self._prepare_data_for_training(tx, rx_real, initial_probs, n_users, num_bits)
        # Training the DeepSICMB network for each user for iteration=1
        train_loss_vect , val_loss_vect = self._train_models(self.detector, 0, tx_all, rx_all, num_bits, n_users, epochs)
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx, num_bits, n_users)
        # Training the DeepSICMB for each user-symbol/iteration
        for i in range(1, iterations):
            # Generating soft symbols for training purposes
            probs_vec, llrs_mat = self._calculate_posteriors(self.detector, i, probs_vec, rx_real.to(device=DEVICE), num_bits, n_users)
            # Obtaining the DeepSICMB networks for each user-symbol and the i-th iteration
            tx_all, rx_all = self._prepare_data_for_training(tx, rx_real.to(device=DEVICE), probs_vec, n_users , num_bits)
            # Training the DeepSICMB networks for the iteration>1
            train_loss_cur , val_loss_cur =  self._train_models(self.detector, i, tx_all, rx_all, num_bits , n_users , epochs)
            if SHOW_ALL_ITERATIONS:
                train_loss_vect = train_loss_vect + train_loss_cur
                val_loss_vect = val_loss_vect + val_loss_cur
        return train_loss_vect , val_loss_vect

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        if tx.dim() == 1:
            tx_rs = tx.unsqueeze(1)
        else:
            tx_rs = tx

        return self.criterion(input=est, target=tx_rs)

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iterations: int, probs_in: torch.Tensor) -> tuple[List, List]:
        # detect and decode
        detected_word_list = [None] * iterations
        llrs_mat_list = [None] * iterations
        probs_vec = self._initialize_probs_for_infer(rx, num_bits, n_users)
        for i in range(iterations):
            probs_vec, llrs_mat_list[i] = self._calculate_posteriors(self.detector, i + 1, probs_vec, rx.to(device=DEVICE), num_bits , n_users)
            detected_word_list[i] = self._compute_output(probs_vec)
        return detected_word_list, llrs_mat_list


    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor, n_users: int, num_bits: int) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(n_users):
            if conf.mod_pilot <= 2:
                idx = [user_i for user_i in range(n_users) if user_i != k]
            else:
                max_value = probs_vec.shape[1]
                all_values = list(range(max_value))

                # Compute the excluded range for the current `i`
                exclude_start = k*num_bits
                exclude_end = (k+1)*num_bits
                # oryeger
                # idx = np.setdiff1d(all_values, range(exclude_start,exclude_end))
                idx = all_values

            kernel_size = conf.kernel_size
            half_kernel = int((conf.kernel_size-1)/2)
            n_probs_per_re = probs_vec.shape[1]
            n_ants_effective_per_re = rx.shape[1]
            probs_vec_extended = torch.zeros(probs_vec.shape[0], n_probs_per_re * kernel_size, conf.num_res).to(DEVICE)
            rx_extended =  torch.zeros(rx.shape[0], n_ants_effective_per_re * kernel_size, conf.num_res).to(DEVICE)
            for re in range(conf.num_res):
                begin_index = re - half_kernel
                end_index = begin_index + kernel_size
                indexes_updated = np.zeros(kernel_size,dtype=int)
                running_idx = 0
                for index in  range(begin_index, end_index):
                    if index<0:
                        indexes_updated[running_idx] = index + kernel_size
                    elif index>conf.num_res-1:
                        indexes_updated[running_idx] = index - kernel_size
                    else:
                        indexes_updated[running_idx] = index
                    running_idx += 1

                running_idx = 0
                for kernel_idx in indexes_updated:
                    probs_vec_extended[:,n_probs_per_re*(running_idx):n_probs_per_re*(running_idx+1),re] = probs_vec[:,:,kernel_idx]
                    rx_extended[:,n_ants_effective_per_re*(running_idx):n_ants_effective_per_re*(running_idx+1),re] = rx[:,:,kernel_idx]
                    running_idx += 1

            current_y_train = torch.cat((rx_extended, probs_vec_extended), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all

    def _initialize_probs_for_training(self, tx, num_bits, n_users):
        num_rows = int(tx.shape[0]//num_bits)
        num_cols = num_bits * n_users
        return HALF * torch.ones(num_rows,num_cols, conf.num_res).to(DEVICE)

    def _initialize_probs(self, tx, num_bits, n_users):
        num_rows = int(tx.shape[0]//num_bits)
        num_cols = num_bits * n_users
        rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(num_rows,num_cols,conf.num_res)))
        return rnd_init

    def _calculate_posteriors(self, model: List[List[List[nn.Module]]], i: int, probs_vec: torch.Tensor,
                              rx: torch.Tensor, num_bits: int, n_users: int) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = probs_vec.clone()
        n_ants_effective_per_re = rx.shape[1]
        kernel_size = conf.kernel_size
        half_kernel = int((conf.kernel_size - 1) / 2)
        for re in range(conf.num_res):
            for user in range(n_users):
                if conf.mod_pilot <= 2:
                    idx = [user_i for user_i in range(n_users) if user_i != user]
                    user_indexes = user
                    local_user_indexes = 0
                else:
                    max_value = probs_vec.shape[1]
                    all_values = list(range(max_value))

                    # Compute the excluded range for the current `i`
                    exclude_start = user*num_bits
                    exclude_end = (user+1)*num_bits
                    # oryeger
                    idx = np.setdiff1d(all_values, range(exclude_start,exclude_end))
                    user_indexes = np.setdiff1d(all_values, idx)
                    # oryeger
                    idx = all_values
                    local_user_indexes = range(0, num_bits)

                n_probs_per_re = probs_vec.shape[1]
                probs_vec_extended = torch.zeros(probs_vec.shape[0], n_probs_per_re * kernel_size).to(DEVICE)
                rx_extended = torch.zeros(rx.shape[0], n_ants_effective_per_re * kernel_size).to(DEVICE)

                begin_index = re - half_kernel
                end_index = begin_index + kernel_size
                indexes_updated = np.zeros(kernel_size,dtype=int)
                running_idx = 0
                for index in  range(begin_index, end_index):
                    if index<0:
                        indexes_updated[running_idx] = index + kernel_size
                    elif index>conf.num_res-1:
                        indexes_updated[running_idx] = index - kernel_size
                    else:
                        indexes_updated[running_idx] = index
                    running_idx += 1

                running_idx = 0
                for kernel_idx in indexes_updated:
                    probs_vec_extended[:, n_probs_per_re * (running_idx):n_probs_per_re * (running_idx + 1)] = probs_vec[:, :, kernel_idx]
                    rx_extended[:, n_ants_effective_per_re * (running_idx):n_ants_effective_per_re * (running_idx + 1)] = rx[:, :, kernel_idx]
                    running_idx += 1

                input = torch.cat((rx_extended, probs_vec_extended), dim=1)
                preprocessed_input = self._preprocess(input)
                with torch.no_grad():
                    output, llrs = model[re][user][i - 1](preprocessed_input)
                next_probs_vec[:, user_indexes,re] = output[:, local_user_indexes]
        return next_probs_vec, llrs

    def _initialize_probs_for_infer(self, rx: torch.Tensor, num_bits: int, n_users: int):
        return HALF * torch.ones(rx.shape[0], n_users*num_bits, conf.num_res).to(DEVICE).float()
