
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.escnn.escnn_detector import ESCNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable

Softmax = torch.nn.Softmax(dim=1)

class ESCNNTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'ESCNN'

    def _initialize_detector(self, num_bits, n_users, n_ants):

        self.detector = [[ESCNNDetector(num_bits, n_users).to(DEVICE) for _ in range(conf.iterations)] for _ in
                         range(n_users)]  # 2D list for Storing the ESCNN Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx_prob: torch.Tensor, num_bits:int, epochs: int, first_half_flag: bool, stage: str) -> list[float]:
        """
        Trains a ESCNN Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)

        if isinstance(single_model, ESCNNDetector):
            single_model.set_stage(stage)

        self._deep_learning_setup(single_model)
        train_loss_vect = []
        val_loss_vect = []

        # Reshape tx to match the expected format
        tx_reshaped = tx.reshape(int(tx.shape[0] // num_bits), num_bits, tx.shape[1])

        # Split into train and validation sets
        train_samples = int(rx_prob.shape[0] * TRAIN_PERCENTAGE / 100)
        rx_prob_train = rx_prob[:train_samples]
        rx_prob_val = rx_prob[train_samples:]
        tx_train = tx_reshaped[:train_samples]
        tx_val = tx_reshaped[train_samples:]

        # Create DataLoader for mini-batch training
        # batch_size <= 0 means full-batch (no mini-batching)
        batch_size = conf.batch_size if hasattr(conf, 'batch_size') else 32
        if batch_size <= 0:
            batch_size = len(rx_prob_train)  # Full batch
        shuffle = conf.shuffle if hasattr(conf, 'shuffle') else True
        train_dataset = TensorDataset(rx_prob_train, tx_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for batch_rx, batch_tx in train_loader:
                batch_rx = batch_rx.to(DEVICE)
                batch_tx = batch_tx.to(DEVICE)

                soft_estimation, llrs = single_model(batch_rx)

                if first_half_flag:
                    llrs_cur = llrs[:, 0::2, :, :]
                    batch_tx_cur = batch_tx[:, 0::2, :]
                else:
                    llrs_cur = llrs
                    batch_tx_cur = batch_tx

                loss = self._calculate_loss(llrs_cur, batch_tx_cur)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_loss_vect.append(avg_train_loss)

            # Calculate validation loss
            with torch.no_grad():
                rx_prob_val_device = rx_prob_val.to(DEVICE)
                _, llrs_val = single_model(rx_prob_val_device)
                if first_half_flag:
                    val_loss = self._calculate_loss(llrs_val[:, 0::2, :, :], tx_val[:, 0::2, :].to(DEVICE))
                else:
                    val_loss = self._calculate_loss(llrs_val, tx_val.to(DEVICE))
                val_loss_vect.append(val_loss.item())

        return train_loss_vect, val_loss_vect

    def _train_models(self, model: List[List[ESCNNDetector]], i: int, tx_all: List[torch.Tensor],
                      rx_prob_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int, first_half_flag: bool, stage: str):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(n_users):
            train_loss_vect , val_loss_vect = self._train_model(model[user][i], tx_all[user], rx_prob_all[user].to(DEVICE), num_bits, epochs, first_half_flag, stage)
            if user == 0:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor, stage: str = "base"):
        """
        Main training function for ESCNN trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        if conf.which_augment == 'NO_AUGMENT':
            initial_probs = self._initialize_probs(tx, num_bits, n_users)
        else:
            initial_probs = probs_in

        # Training the ESCNN network for each user for iteration=1
        tx_all, rx_prob_all = self._prepare_data_for_training(tx, rx_real, initial_probs, n_users)
        train_loss_vect , val_loss_vect = self._train_models(self.detector, 0, tx_all, rx_prob_all, num_bits, n_users, epochs, first_half_flag, stage)
        # Initializing the probabilities
        if conf.which_augment == 'NO_AUGMENT':
            probs_vec = self._initialize_probs_for_training(tx, num_bits, n_users)
        else:
            probs_vec = probs_in.to(DEVICE)
        # Training the ESCNN for each user-symbol/iteration
        for i in range(1, iterations):
            # Training the ESCNN networks for the iteration>1
            # Generating soft symbols for training purposes
            probs_vec, llrs_mat = self._calculate_posteriors(self.detector, i, rx_real.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits,n_users, 0)
            tx_all, rx_prob_all = self._prepare_data_for_training(tx, rx_real.to(device=DEVICE), probs_vec, n_users)
            train_loss_cur , val_loss_cur =  self._train_models(self.detector, i, tx_all, rx_prob_all, num_bits, n_users, epochs, first_half_flag, stage)
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
        detected_word_list = [None] * iterations
        llrs_mat_list = [None] * iterations
        if conf.which_augment == 'NO_AUGMENT':
            probs_vec = self._initialize_probs_for_infer(rx, num_bits, n_users)
        else:
            probs_vec = probs_in.to(DEVICE)

        nns = 0
        for i in range(iterations):
            probs_vec, llrs_mat_list[i] = self._calculate_posteriors(self.detector, i + 1, rx.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits, n_users, nns)
            detected_word_list[i] = self._compute_output(probs_vec)
        # plt.imshow(self.detector[0][0].fc1.weight[0, :, 0, :].cpu().detach(), cmap='gray')
        # pass

        return detected_word_list, llrs_mat_list



    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor, n_users: int) -> [torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_prob_all = []
        for user in range(n_users):
            if conf.no_probs:
                rx_prob_all.append(rx.unsqueeze(-1))
            else:
                rx_prob_all.append(torch.cat((rx.unsqueeze(-1), probs_vec), dim=1))
            tx_all.append(tx[:, user, :])
        return tx_all, rx_prob_all

    def _initialize_probs_for_training(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)

    def _initialize_probs(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        # rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(dim0,dim1,dim2,dim3)).astype(np.float32))
        rnd_init = HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32)
        return rnd_init

    def _calculate_posteriors(self, model: List[List[nn.Module]], i: int, rx_real: torch.Tensor, prob: torch.tensor, num_bits: int, n_users: int, nns: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = prob.clone()
        # next_probs_vec = torch.zeros_like(prob)
        llrs_mat = torch.zeros(next_probs_vec.shape).to(DEVICE)
        for user in range(n_users):
            if conf.no_probs:
                rx_prob = rx_real
            else:
                rx_prob = torch.cat((rx_real, prob), dim=1)

            with torch.no_grad():
                output, llrs = model[user][i - 1](rx_prob)
            index_start = user * num_bits
            index_end = (user + 1) * num_bits
            next_probs_vec[:, index_start:index_end, :, :] = output
            llrs_mat[:, index_start:index_end, :, :] = llrs

        return next_probs_vec, llrs_mat

    def _initialize_probs_for_infer(self, rx: torch.Tensor, num_bits: int, n_users: int):
        dim0 = rx.shape[0]
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)