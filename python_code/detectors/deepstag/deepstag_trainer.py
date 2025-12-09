from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deepstag.deepstag_det_re import DeepSTAGDetRe
from python_code.detectors.deepstag.deepstag_det_conv import DeepSTAGDetConv
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable
import numpy as np

Softmax = torch.nn.Softmax(dim=1)

class DeepSTAGTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        self.lr = 5e-3
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'DeepSTAG'

    def _initialize_detector(self, num_bits, n_users, n_ants):

        self.det_re = [[[DeepSTAGDetRe(num_bits, n_users).to(DEVICE) for _ in range(conf.iterations )] for _ in range(n_users)] for _ in
                         range(conf.num_res)]  # 2D list for Storing the DeepSTAG Networks

        self.det_conv = [[DeepSTAGDetConv(num_bits, n_users).to(DEVICE) for _ in range(conf.iterations)] for _ in range(n_users)]  # 2D list for Storing the DeepSTAG Networks

    def _train_models_re(self, model: List[List[List[DeepSTAGDetRe]]], i: int, tx_all: List[torch.Tensor], rx_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for re in range(conf.num_res):
            for user in range(n_users):
                train_loss_vect , val_loss_vect = self._train_model_re(model[re][user][i], tx_all[user][:,re], rx_all[user][:,:,re].to(DEVICE), num_bits, epochs)
                if user == 0:
                    train_loss_vect_user = train_loss_vect
                    val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user


    def _train_model_re(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor, num_bits:int, epochs: int) -> list[float]:
        """
        Trains a DeepSTAG Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        y_total = self._preprocess(rx)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        for _ in range(epochs):
            soft_estimation, llrs = single_model(y_total)
            if num_bits <= 1:
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

    def _train_models_conv(self, model: List[List[DeepSTAGDetConv]], i: int, tx_all: List[torch.Tensor],
                      rx_prob_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(n_users):
            train_loss_vect , val_loss_vect = self._train_model_conv(model[user][i], tx_all[user], rx_prob_all[user].to(DEVICE), num_bits, epochs)
            if user == 1:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user

    def _train_model_conv(self, single_model: nn.Module, tx: torch.Tensor, rx_prob: torch.Tensor, num_bits:int, epochs: int) -> list[float]:
        """
        Trains a DeepSTAG Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        # for _ in range(epochs):
        for epoch in range(epochs):
            soft_estimation, llrs = single_model(rx_prob)
            tx_cur = tx
            tx_reshaped = tx_cur.reshape(int(tx_cur.shape[0] // num_bits), num_bits, tx_cur.shape[1])

            train_samples = int(soft_estimation.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop(soft_estimation[:train_samples], tx_reshaped[:train_samples], False)
            val_loss = self._calculate_loss(soft_estimation[train_samples:], tx_reshaped[train_samples:])
            val_loss = val_loss.item()
            loss += current_loss
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect


    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor):
        """
        Main training function for DeepSTAG trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        # Training the DeepSTAG network for each user for iteration=1
        initial_probs = self._initialize_probs_conv(tx, num_bits, n_users).to(device=DEVICE)
        tx_all, rx_prob_all = self._prepare_data_for_training_conv(tx, rx_real.to(device=DEVICE), initial_probs.squeeze(-1), n_users)
        train_loss_vect, val_loss_vect = self._train_models_conv(self.det_conv, 0, tx_all, rx_prob_all, num_bits, n_users,epochs)
        probs_vec = self._initialize_probs_for_training_conv(tx, num_bits, n_users).to(device=DEVICE)
        probs_vec, llrs_mat = self._calculate_posteriors_conv(self.det_conv, 1, rx_real.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits, n_users)
        tx_all, rx_all = self._prepare_data_for_training_re(tx, rx_real, probs_vec.squeeze(-1), n_users)
        train_loss_cur, val_loss_cur = self._train_models_re(self.det_re, 0, tx_all, rx_all, num_bits, n_users,epochs)
        train_loss_vect = train_loss_vect + train_loss_cur
        val_loss_vect = val_loss_vect + val_loss_cur

        if iterations>1:
            probs_vec, llrs_mat = self._calculate_posteriors_re(self.det_re, 1, rx_real.to(device=DEVICE),
                                                                probs_vec.squeeze(-1), num_bits, n_users)        # Initializing the probabilities
        # Training the DeepSTAGNet for each user-symbol/iteration
        for i in range(1, iterations):
            # Training the DeepSTAG networks for the iteration>1
            tx_all, rx_prob_all = self._prepare_data_for_training_conv(tx, rx_real.to(device=DEVICE), probs_vec,n_users)
            train_loss_cur, train_loss_cur = self._train_models_conv(self.det_conv,i, tx_all, rx_prob_all, num_bits,n_users, epochs)
            if SHOW_ALL_ITERATIONS:
                train_loss_vect = train_loss_vect + train_loss_cur
                val_loss_vect = val_loss_vect + train_loss_cur

            probs_vec, _ = self._calculate_posteriors_conv(self.det_conv, i + 1,
                                                           rx_real.to(device=DEVICE).unsqueeze(-1),
                                                           probs_vec.unsqueeze(-1), num_bits, n_users)
            tx_all, rx_all = self._prepare_data_for_training_re(tx, rx_real, probs_vec.squeeze(-1), n_users)
            train_loss_cur, val_loss_cur = self._train_models_re(self.det_re, i, tx_all, rx_all, num_bits, n_users,epochs)
            if i != iterations-1:
                probs_vec, llrs_mat = self._calculate_posteriors_re(self.det_re, i+1, rx_real.to(device=DEVICE),
                                                                    probs_vec.squeeze(-1), num_bits,
                                                                    n_users)  # Initializing the probabilities
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

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iterations: int, probs_in: torch.Tensor) -> tuple[List, List]:
        # detect and decode
        detected_word_list = [None] * iterations * 2
        llrs_mat_list = [None] * iterations * 2
        probs_vec = self._initialize_probs_for_infer_conv(rx, num_bits, n_users)
        for i in range(iterations):
            probs_vec, llrs_mat_list[i*2] = self._calculate_posteriors_conv(self.det_conv, i + 1, rx.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits, n_users)
            detected_word_list[i*2] = self._compute_output(probs_vec)
            probs_vec, llrs_mat_list[i*2+1] = self._calculate_posteriors_re(self.det_re, i + 1, rx.to(device=DEVICE), probs_vec.squeeze(-1), num_bits, n_users)
            probs_vec = probs_vec.unsqueeze(-1)
            detected_word_list[i*2+1] = self._compute_output(probs_vec)

        return detected_word_list, llrs_mat_list



    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training_re(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor, n_users: int) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_all = []
        for k in range(n_users):
            max_value = probs_vec.shape[1]
            all_values = list(range(max_value))
            idx = all_values

            kernel_size = conf.stag_re_kernel_size
            half_kernel = int((kernel_size-1)/2)
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


    def _prepare_data_for_training_conv(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor, n_users: int) -> [torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_prob_all = []
        for user in range(n_users):
            rx_prob_all.append(torch.cat((rx.unsqueeze(-1), probs_vec.unsqueeze(-1)), dim=1))
            tx_all.append(tx[:, user, :])
        return tx_all, rx_prob_all

    def _initialize_probs_for_training_re(self, tx, num_bits, n_users):
        num_rows = int(tx.shape[0]//num_bits)
        num_cols = num_bits * n_users
        return HALF * torch.ones(num_rows,num_cols, conf.num_res).to(DEVICE)

    def _initialize_probs_for_training_conv(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)

    def _initialize_probs_re(self, tx, num_bits, n_users):
        num_rows = int(tx.shape[0]//num_bits)
        num_cols = num_bits * n_users
        rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(num_rows,num_cols,conf.num_res)))
        return rnd_init

    def _initialize_probs_conv(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        # rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(dim0,dim1,dim2,dim3)).astype(np.float32))
        rnd_init = HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32)
        return rnd_init

    def _calculate_posteriors_re(self, model: List[List[List[DeepSTAGDetRe]]], i: int, rx: torch.Tensor, probs_vec: torch.Tensor,
                               num_bits: int, n_users: int) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = probs_vec.clone()
        n_ants_effective_per_re = rx.shape[1]
        kernel_size = conf.stag_re_kernel_size
        half_kernel = int((kernel_size - 1) / 2)
        for re in range(conf.num_res):
            for user in range(n_users):
                max_value = probs_vec.shape[1]
                all_values = list(range(max_value))

                # Compute the excluded range for the current `i`
                exclude_start = user*num_bits
                exclude_end = (user+1)*num_bits
                idx = np.setdiff1d(all_values, range(exclude_start,exclude_end))
                user_indexes = np.setdiff1d(all_values, idx)
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

    def _calculate_posteriors_conv(self, model: List[List[DeepSTAGDetConv]], i: int, rx_real: torch.Tensor, prob: torch.tensor, num_bits: int, n_users: int) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = prob.clone()
        # next_probs_vec = torch.zeros_like(prob)
        llrs_mat = torch.zeros(next_probs_vec.shape).to(DEVICE)
        for user in range(n_users):
                rx_prob = torch.cat((rx_real, prob), dim=1)

                with torch.no_grad():
                    output, llrs = model[user][i - 1](rx_prob)
                index_start = user * num_bits
                index_end = (user + 1) * num_bits
                next_probs_vec[:, index_start:index_end, :, :] = output
                llrs_mat[:, index_start:index_end, :, :] = llrs

        return next_probs_vec, llrs_mat

    def _initialize_probs_for_infer_re(self, rx: torch.Tensor, num_bits: int, n_users: int):
        return HALF * torch.ones(rx.shape[0], n_users*num_bits, conf.num_res).to(DEVICE).float()

    def _initialize_probs_for_infer_conv(self, rx: torch.Tensor, num_bits: int, n_users: int):
        dim0 = rx.shape[0]
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)