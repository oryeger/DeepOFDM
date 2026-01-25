from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.jointllr.jointllr_detector import JointLLRDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS


class JointLLRTrainer(Trainer):
    """
    Trainer for the JointLLR detector.

    Key differences from ESCNN:
    - Processes all users jointly (single network outputs all users' bits)
    - Uses channel matrix H as explicit conditioning (FiLM)
    - Supports LLR feedback for iterative refinement
    """

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'JointLLR'

    def _initialize_detector(self, num_bits, n_users, n_ants):
        """
        Initialize detector(s) for each iteration.
        Unlike ESCNN which has [n_users][iterations] networks,
        JointLLR has [iterations] networks (one per iteration, processing all users jointly).
        """
        self.detector = [JointLLRDetector(num_bits, n_users, n_ants).to(DEVICE)
                         for _ in range(conf.iterations)]

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor,
                     y: torch.Tensor, H: torch.Tensor, prior_llrs: torch.Tensor,
                     num_bits: int, epochs: int) -> Tuple[List[float], List[float]]:
        """
        Trains a JointLLR Network and returns the training/validation loss.

        Args:
            single_model: The detector model to train
            tx: Ground truth bits (batch, n_users*num_bits, num_res)
            y: Received signal (batch, 2*n_ants, num_res)
            H: Channel matrix (batch, 2*n_ants*n_users, num_res)
            prior_llrs: Prior LLRs (batch, n_users*num_bits, num_res) or None
            num_bits: Number of bits per symbol
            epochs: Number of training epochs
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        train_loss_vect = []
        val_loss_vect = []

        # Reshape inputs for the network: (batch, n_symbols, features)
        # y: (batch, 2*n_ants, num_res) -> (batch, num_res, 2*n_ants)
        # H: (batch, 2*n_ants*n_users, num_res) -> (batch, num_res, 2*n_ants*n_users)
        # prior_llrs: (batch, n_users*num_bits, num_res) -> (batch, num_res, n_users*num_bits)
        y_transposed = y.permute(0, 2, 1)  # (batch, num_res, 2*n_ants)
        H_transposed = H.permute(0, 2, 1)  # (batch, num_res, 2*n_ants*n_users)

        if prior_llrs is not None:
            prior_transposed = prior_llrs.permute(0, 2, 1)  # (batch, num_res, n_users*num_bits)
        else:
            prior_transposed = None

        # tx: (batch*num_bits, n_users, num_res) needs to be reshaped to match output
        # Output is (batch, num_res, n_users*num_bits)
        # tx is structured as (n_symbols*num_bits, n_users, num_res)
        n_symbols = y_transposed.shape[0]
        n_users = conf.n_users

        # Reshape tx to (n_symbols, num_res, n_users*num_bits)
        # tx input is (n_symbols*num_bits, n_users, num_res)
        tx_reshaped = tx.reshape(n_symbols, num_bits, n_users, conf.num_res)
        tx_reshaped = tx_reshaped.permute(0, 3, 2, 1)  # (n_symbols, num_res, n_users, num_bits)
        tx_reshaped = tx_reshaped.reshape(n_symbols, conf.num_res, n_users * num_bits)

        # Split into train and validation sets
        train_samples = int(n_symbols * TRAIN_PERCENTAGE / 100)

        y_train = y_transposed[:train_samples]
        y_val = y_transposed[train_samples:]
        H_train = H_transposed[:train_samples]
        H_val = H_transposed[train_samples:]
        tx_train = tx_reshaped[:train_samples]
        tx_val = tx_reshaped[train_samples:]

        if prior_transposed is not None:
            prior_train = prior_transposed[:train_samples]
            prior_val = prior_transposed[train_samples:]
        else:
            prior_train = None
            prior_val = None

        # Create DataLoader for mini-batch training
        batch_size = conf.batch_size if hasattr(conf, 'batch_size') else 32
        if batch_size <= 0:
            batch_size = len(y_train)
        shuffle = conf.shuffle if hasattr(conf, 'shuffle') else True

        if prior_train is not None:
            train_dataset = TensorDataset(y_train, H_train, prior_train, tx_train)
        else:
            # Create dummy priors of zeros
            dummy_prior = torch.zeros(y_train.shape[0], conf.num_res, n_users * num_bits,
                                      dtype=torch.float32)
            train_dataset = TensorDataset(y_train, H_train, dummy_prior, tx_train)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch training
            for batch_data in train_loader:
                batch_y, batch_H, batch_prior, batch_tx = batch_data
                batch_y = batch_y.to(DEVICE)
                batch_H = batch_H.to(DEVICE)
                batch_prior = batch_prior.to(DEVICE)
                batch_tx = batch_tx.to(DEVICE)

                probs, llrs = single_model(batch_y, batch_H, batch_prior)

                loss = self._calculate_loss(llrs, batch_tx)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_loss_vect.append(avg_train_loss)

            # Calculate validation loss
            with torch.no_grad():
                y_val_device = y_val.to(DEVICE)
                H_val_device = H_val.to(DEVICE)
                if prior_val is not None:
                    prior_val_device = prior_val.to(DEVICE)
                else:
                    prior_val_device = torch.zeros(y_val.shape[0], conf.num_res, n_users * num_bits,
                                                   dtype=torch.float32, device=DEVICE)

                _, llrs_val = single_model(y_val_device, H_val_device, prior_val_device)
                val_loss = self._calculate_loss(llrs_val, tx_val.to(DEVICE))
                val_loss_vect.append(val_loss.item())

        return train_loss_vect, val_loss_vect

    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor,
                         H_all: torch.Tensor, num_bits: int, n_users: int,
                         iterations: int, epochs: int, first_half_flag: bool,
                         probs_in: torch.Tensor) -> Tuple[List[float], List[float]]:
        """
        Main training function for JointLLR trainer.

        Args:
            tx: Transmitted bits
            rx_real: Received signal (real/imag interleaved)
            H_all: Channel matrix for all REs (n_symbols, 2*n_ants*n_users, num_res)
            num_bits: Number of bits per symbol
            n_users: Number of users
            iterations: Number of iterative refinement passes
            epochs: Training epochs per iteration
            first_half_flag: Whether to train on first half only (for special modulation)
            probs_in: Initial probabilities from primary detector (or None)
        """
        # Convert probs to LLRs if provided
        if conf.which_augment == 'NO_AUGMENT' or probs_in is None or probs_in.numel() == 0:
            initial_llrs = None
        else:
            # probs_in shape: (n_symbols, n_users*num_bits, num_res, 1)
            probs_squeezed = probs_in.squeeze(-1)  # (n_symbols, n_users*num_bits, num_res)
            # Convert probs to LLRs: LLR = log(p / (1-p))
            eps = 1e-7
            probs_clamped = torch.clamp(probs_squeezed, eps, 1 - eps)
            initial_llrs = torch.log(probs_clamped / (1 - probs_clamped))

        # Train first iteration detector
        train_loss_vect, val_loss_vect = self._train_model(
            self.detector[0], tx, rx_real, H_all, initial_llrs, num_bits, epochs)

        # For subsequent iterations, use output from previous iteration as prior
        current_llrs = initial_llrs
        for i in range(1, iterations):
            # Get LLRs from previous iteration's detector
            current_llrs = self._get_llrs_for_training(
                self.detector[i-1], rx_real, H_all, current_llrs, num_bits, n_users)

            # Train current iteration's detector
            train_loss_cur, val_loss_cur = self._train_model(
                self.detector[i], tx, rx_real, H_all, current_llrs, num_bits, epochs)

            if SHOW_ALL_ITERATIONS:
                train_loss_vect = train_loss_vect + train_loss_cur
                val_loss_vect = val_loss_vect + val_loss_cur

        return train_loss_vect, val_loss_vect

    def _get_llrs_for_training(self, model: nn.Module, rx_real: torch.Tensor,
                                H_all: torch.Tensor, prior_llrs: torch.Tensor,
                                num_bits: int, n_users: int) -> torch.Tensor:
        """
        Run inference to get LLRs for training the next iteration.
        """
        with torch.no_grad():
            # Reshape inputs
            y_transposed = rx_real.permute(0, 2, 1).to(DEVICE)
            H_transposed = H_all.permute(0, 2, 1).to(DEVICE)

            if prior_llrs is not None:
                prior_transposed = prior_llrs.permute(0, 2, 1).to(DEVICE)
            else:
                prior_transposed = torch.zeros(
                    y_transposed.shape[0], conf.num_res, n_users * num_bits,
                    dtype=torch.float32, device=DEVICE)

            _, llrs = model(y_transposed, H_transposed, prior_transposed)

            # Transpose back: (batch, num_res, n_users*num_bits) -> (batch, n_users*num_bits, num_res)
            return llrs.permute(0, 2, 1)

    def _calculate_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
        Binary Cross Entropy loss with logits.
        est: (batch, num_res, n_users*num_bits) - LLRs
        tx: (batch, num_res, n_users*num_bits) - ground truth bits
        """
        return self.criterion(input=est, target=tx)

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, rx: torch.Tensor, H_all: torch.Tensor, num_bits: int,
                 n_users: int, iterations: int, probs_in: torch.Tensor) -> Tuple[List, List]:
        """
        Forward pass for inference.

        Args:
            rx: Received signal (n_symbols, 2*n_ants, num_res)
            H_all: Channel matrix (n_symbols, 2*n_ants*n_users, num_res)
            num_bits: Number of bits per symbol
            n_users: Number of users
            iterations: Number of refinement iterations
            probs_in: Initial probabilities from primary detector

        Returns:
            detected_word_list: List of detected words per iteration
            llrs_mat_list: List of LLR matrices per iteration
        """
        detected_word_list = [None] * iterations
        llrs_mat_list = [None] * iterations

        # Convert probs to LLRs if provided
        if conf.which_augment == 'NO_AUGMENT' or probs_in is None or probs_in.numel() == 0:
            current_llrs = None
        else:
            probs_squeezed = probs_in.squeeze(-1)
            eps = 1e-7
            probs_clamped = torch.clamp(probs_squeezed, eps, 1 - eps)
            current_llrs = torch.log(probs_clamped / (1 - probs_clamped))

        # Reshape inputs
        y_transposed = rx.permute(0, 2, 1).to(DEVICE)  # (batch, num_res, 2*n_ants)
        H_transposed = H_all.permute(0, 2, 1).to(DEVICE)  # (batch, num_res, 2*n_ants*n_users)

        for i in range(iterations):
            if current_llrs is not None:
                prior_transposed = current_llrs.permute(0, 2, 1).to(DEVICE)
            else:
                prior_transposed = torch.zeros(
                    y_transposed.shape[0], conf.num_res, n_users * num_bits,
                    dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                probs, llrs = self.detector[i](y_transposed, H_transposed, prior_transposed)

            # Transpose back: (batch, num_res, features) -> (batch, features, num_res)
            probs_out = probs.permute(0, 2, 1)  # (batch, n_users*num_bits, num_res)
            llrs_out = llrs.permute(0, 2, 1)    # (batch, n_users*num_bits, num_res)

            # Update current_llrs for next iteration
            current_llrs = llrs_out

            # Store results with extra dimension for compatibility
            llrs_mat_list[i] = llrs_out.unsqueeze(-1)  # (batch, n_users*num_bits, num_res, 1)
            detected_word_list[i] = self._compute_output(probs_out)

        return detected_word_list, llrs_mat_list

    def _compute_output(self, probs_vec: torch.Tensor) -> torch.Tensor:
        """
        Convert probabilities to detected bits.
        probs_vec: (batch, n_users*num_bits, num_res)
        """
        # Add extra dimension for compatibility with existing code
        probs_expanded = probs_vec.unsqueeze(-1)  # (batch, n_users*num_bits, num_res, 1)
        symbols_word = prob_to_BPSK_symbol(probs_expanded.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word
