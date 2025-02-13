from typing import List

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deepsic.deepsic_detector import DeepSICDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, N_USERS, N_ANTS, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol, prob_to_QAM_index
import numpy as np
from python_code.utils.constants import MOD_PILOT, SHOW_ALL_ITERATIONS, EPOCHS, NUM_BITS
import commpy.modulation as mod

ITERATIONS = 5

Softmax = torch.nn.Softmax(dim=1)

class DeepSICTrainer(Trainer):

    def __init__(self):
        self.lr = 5e-3
        super().__init__()

    def __str__(self):
        return 'DeepSIC'

    def _initialize_detector(self):
        self.detector = [[DeepSICDetector().to(DEVICE) for _ in range(ITERATIONS)] for _ in
                         range(N_USERS*conf.num_res)]  # 2D list for Storing the DeepSIC Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx: torch.Tensor, prob: torch.Tensor) -> list[float]:
        """
        Trains a DeepSIC Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        y_total = self._preprocess(rx)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        for _ in range(EPOCHS):
            soft_estimation = single_model(prob, rx)
            if MOD_PILOT <= 2:
                tx_reshaped = tx
            else:
                tx_reshaped = tx.reshape(int(tx.numel() // NUM_BITS), NUM_BITS)

            train_samples = int(soft_estimation.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop(soft_estimation[:train_samples], tx_reshaped[:train_samples])
            val_loss = self._calculate_loss(soft_estimation[train_samples:], tx_reshaped[train_samples:])
            val_loss = val_loss.item()
            loss += current_loss
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect

    def _train_models(self, model: List[List[DeepSICDetector]], i: int, tx_all: List[torch.Tensor],
                      rx: List[torch.Tensor], prob_all: List[torch.Tensor]):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(N_USERS):
            train_loss_vect , val_loss_vect = self._train_model(model[user][i], tx_all[user], rx, prob_all[user])
            if user == 0:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor):
        """
        Main training function for DeepSIC trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        initial_probs = self._initialize_probs(tx)
        tx_all, prob_all = self._prepare_data_for_training(tx, rx_real, initial_probs)
        # Training the DeepSIC network for each user for iteration=1
        train_loss_vect , val_loss_vect = self._train_models(self.detector, 0, tx_all, rx_real, prob_all)
        # Initializing the probabilities
        probs_vec = self._initialize_probs_for_training(tx)
        # Training the DeepSICNet for each user-symbol/iteration
        for i in range(1, ITERATIONS):
            # Generating soft symbols for training purposes
            probs_vec = self._calculate_posteriors(self.detector, i, probs_vec, rx_real)
            # Obtaining the DeepSIC networks for each user-symbol and the i-th iteration
            tx_all, prob_all = self._prepare_data_for_training(tx, rx_real, probs_vec)
            # Training the DeepSIC networks for the iteration>1
            train_loss_cur , val_loss_cur =  self._train_models(self.detector, i, tx_all, rx_real, prob_all)
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

    def _forward(self, rx: torch.Tensor) -> torch.Tensor:
        # detect and decode
        probs_vec = self._initialize_probs_for_infer(rx)
        for i in range(ITERATIONS):
            probs_vec = self._calculate_posteriors(self.detector, i + 1, probs_vec, rx)
        return self._compute_output(probs_vec)



    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        prob_all = []
        for user in range(N_USERS):
            max_value = probs_vec.shape[1]
            all_values = list(range(max_value))

            # Compute the excluded range for the current `i`
            # exclude_start = (conf.num_res*user + re)*NUM_BITS
            # exclude_end = (conf.num_res*user + re+1)*NUM_BITS
            # idx = np.setdiff1d(all_values, range(exclude_start,exclude_end))
            idx = all_values

            prob_all.append(probs_vec)
            tx_all.append(tx[:, user, :])
        return tx_all, prob_all

    def _initialize_probs_for_training(self, tx):
        dim0 = int(tx.shape[0]//NUM_BITS)
        dim1 = NUM_BITS * N_USERS
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)

    def _initialize_probs(self, tx):
        dim0 = int(tx.shape[0]//NUM_BITS)
        dim1 = NUM_BITS * N_USERS
        dim2 = conf.num_res
        dim3 = 1
        rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(dim0,dim1,dim2,dim3)).astype(np.float32))
        return rnd_init

    def _calculate_posteriors(self, model: List[List[nn.Module]], i: int, probs_vec: torch.Tensor,
                              rx: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = torch.zeros(probs_vec.shape).to(DEVICE)
        for user in range(N_USERS):
            local_user_indexes = range(0, NUM_BITS)
            with torch.no_grad():
                output = model[user][i - 1](probs_vec, rx)
            index_start = user*NUM_BITS
            index_end = (user+1) * NUM_BITS
            output = output.view(next_probs_vec.shape[0],NUM_BITS,next_probs_vec.shape[2],next_probs_vec.shape[3])
            next_probs_vec[:, index_start:index_end,:,:] = output
        return next_probs_vec

    def _initialize_probs_for_infer(self, rx: torch.Tensor):
        return HALF * torch.ones(rx.shape[0], N_USERS*NUM_BITS*conf.num_res).to(DEVICE).float() # was N_ANTS instead of N_USERS
