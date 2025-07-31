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
                         range(conf.num_res)]  # 2D list for Storing the DeepSIC Networks

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
        Trains a DeepSIC Network and returns the total training loss.
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

    def _train_models_conv(self, model: List[List[List[DeepSTAGDetConv]]], i: int, tx_all: List[torch.Tensor],
                      rx_prob_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(n_users):
            train_loss_vect , val_loss_vect = self._train_model_conv(model[user][i], tx_all[user], rx_prob_all[user].to(DEVICE), num_bits, epochs)
            if user == 3:
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


    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool):
        """
        Main training function for DeepSTAG trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        # Training the DeepSTAG network for each user for iteration=1



        initial_probs = self._initialize_probs_re(tx, num_bits, n_users)
        tx_all, rx_all = self._prepare_data_for_training_re(tx, rx_real, initial_probs, n_users)
        train_loss_vect , val_loss_vect = self._train_models_re(self.det_re, 0, tx_all, rx_all, num_bits, n_users, epochs)
        probs_vec = self._initialize_probs_for_training_re(tx, num_bits, n_users)
        probs_vec, llrs_mat = self._calculate_posteriors_re(self.det_re, 1,  rx_real.to(device=DEVICE), probs_vec, num_bits, n_users)
        tx_all, rx_prob_all = self._prepare_data_for_training_conv(tx, rx_real.to(device=DEVICE), probs_vec, n_users, num_bits)
        train_loss_cur, val_loss_cur = self._train_models_conv(self.det_conv, 0, tx_all, rx_prob_all, num_bits, n_users,
                                                            epochs)
        train_loss_vect = train_loss_vect + train_loss_cur
        val_loss_vect = val_loss_vect + val_loss_cur

        if iterations>1:
            probs_vec, llrs_mat = self._calculate_posteriors_conv(self.det_conv, 1, rx_real.to(device=DEVICE).unsqueeze(-1),
                                                             probs_vec.unsqueeze(-1), num_bits, n_users)
        # Initializing the probabilities
        # Training the DeepSTAGNet for each user-symbol/iteration
        for i in range(1, iterations):
            # Training the DeepSTAG networks for the iteration>1
            tx_all, rx_all = self._prepare_data_for_training_re(tx, rx_real.to(device=DEVICE), probs_vec.squeeze(-1), n_users)
            train_loss_cur, val_loss_cur = self._train_models_re(self.det_re, i, tx_all, rx_all, num_bits, n_users,
                                                                   epochs)
            if SHOW_ALL_ITERATIONS:
                train_loss_vect = train_loss_vect + train_loss_cur
                val_loss_vect = val_loss_vect + val_loss_cur

            probs_vec, llrs_mat = self._calculate_posteriors_re(self.det_re, i+1, rx_real.to(device=DEVICE), probs_vec.squeeze(-1), num_bits, n_users)
            tx_all, rx_prob_all = self._prepare_data_for_training_conv(tx, rx_real.to(device=DEVICE), probs_vec,n_users, num_bits)
            train_loss_cur, val_loss_cur = self._train_models_conv(self.det_conv, i, tx_all, rx_prob_all, num_bits,n_users,epochs)
            if i != iterations-1:
                probs_vec, llrs_mat = self._calculate_posteriors_conv(self.det_conv, i+1,
                                                                      rx_real.to(device=DEVICE).unsqueeze(-1),
                                                                      probs_vec.unsqueeze(-1), num_bits, n_users)
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

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iterations: int) -> tuple[List, List]:
        # detect and decode
        detected_word_list = [None] * iterations
        llrs_mat_list = [None] * iterations
        probs_vec = self._initialize_probs_for_infer_re(rx, num_bits, n_users)
        for i in range(iterations):
            probs_vec, _ = self._calculate_posteriors_re(self.det_re, i + 1, rx.to(device=DEVICE), probs_vec, num_bits, n_users)
            probs_vec, llrs_mat_list[i] = self._calculate_posteriors_conv(self.det_conv, i + 1, rx.to(device=DEVICE).unsqueeze(-1), probs_vec.unsqueeze(-1), num_bits, n_users)
            probs_vec = probs_vec.squeeze(-1)
            detected_word_list[i] = self._compute_output(probs_vec)

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
            if conf.mod_pilot <= 2:
                idx = [user_i for user_i in range(n_users) if user_i != k]
            else:
                max_value = probs_vec.shape[1]
                all_values = list(range(max_value))
                idx = all_values

            current_y_train = torch.cat((rx, probs_vec[:, idx]), dim=1)
            tx_all.append(tx[:, k])
            rx_all.append(current_y_train)
        return tx_all, rx_all


    def _prepare_data_for_training_conv(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor, n_users: int, num_bits: int) -> [torch.Tensor, torch.Tensor]:
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

    def _calculate_posteriors_re(self, model: List[List[List[nn.Module]]], i: int, rx: torch.Tensor, probs_vec: torch.Tensor,
                               num_bits: int, n_users: int) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = probs_vec.clone()
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


                input = torch.cat((rx[:,:,re], probs_vec[:,idx,re]), dim=1)
                preprocessed_input = self._preprocess(input)
                with torch.no_grad():
                    output, llrs = model[re][user][i - 1](preprocessed_input)
                next_probs_vec[:, user_indexes,re] = output[:, local_user_indexes]
        return next_probs_vec, llrs

    def _calculate_posteriors_conv(self, model: List[List[List[nn.Module]]], i: int, rx_real: torch.Tensor, prob: torch.tensor, num_bits: int, n_users: int) -> torch.Tensor:
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
