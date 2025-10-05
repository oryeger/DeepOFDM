from typing import List, Tuple

import torch
from torch import nn

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.detectors.deepsice2e.deepsice2e_detector import DeepSICe2eDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable


Softmax = torch.nn.Softmax(dim=1)

class DeepSICe2eTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        self.lr = 5e-3
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'DeepSICe2e'

    def _initialize_detector(self, num_bits, n_users, n_ants):

        if conf.full_e2e:
            self.detector = [DeepSICe2eDetector(num_bits, n_users).to(DEVICE)  for _ in  range(1)]   # 2D list for Storing the DeepSICe2e Networks
        else:
            self.detector = [DeepSICe2eDetector(num_bits, n_users).to(DEVICE)  for _ in  range(conf.iters_e2e)]  # 2D list for Storing the DeepSICe2e Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx_prob: torch.Tensor, num_bits: int, epochs: int, iters_vec: torch.Tensor) -> Tuple[List[float], List[float]]:
        """
        Trains a DeepSICe2e Network and returns the total training loss.
        """
        single_model = single_model.to(DEVICE)
        self._deep_learning_setup(single_model)
        loss = 0
        train_loss_vect = []
        val_loss_vect = []
        for _ in range(epochs):
            soft_estimation, llrs = single_model(rx_prob,num_bits,iters_vec)
            train_samples = int(soft_estimation.shape[0]*TRAIN_PERCENTAGE/100)
            current_loss = self.run_train_loop(soft_estimation[:train_samples], tx[:train_samples].to(DEVICE),False)
            val_loss = self._calculate_loss(soft_estimation[train_samples:], tx[train_samples:].to(DEVICE))
            val_loss = val_loss.item()
            loss += current_loss
            train_loss_vect.append(current_loss)
            val_loss_vect.append(val_loss)
        return train_loss_vect , val_loss_vect

    def _train_models(self, model: List[DeepSICe2eDetector], tx_cur: torch.Tensor, rx_prob: torch.Tensor, num_bits: int, n_users: int, epochs: int):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(n_users):
            index_start = user*num_bits
            index_end = (user+1)*num_bits
            tx_cur_user = tx_cur[:,index_start:index_end,:]
            train_loss_vect , val_loss_vect = self._train_model(model[user], tx_cur_user, rx_prob.to(DEVICE), num_bits, epochs, user)
            if user == 0:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user


    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iters_e2e: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor):
        """
        Main training function for DeepSICe2e trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        initial_probs = self._initialize_probs(tx, num_bits, n_users)

        # Training the DeepSICe2e network for each user for iteration=1
        tx_cur = torch.zeros(int(tx.shape[0]/num_bits), num_bits*tx.shape[1], tx.shape[2])
        for i in range(rx_real.shape[0]):
            for user in range(n_users):
                index_start_get = i * num_bits
                index_end_get = (i + 1) * num_bits
                index_start_put = user * num_bits
                index_end_put = (user + 1) * num_bits
                tx_cur[i, index_start_put:index_end_put, :] = tx[index_start_get:index_end_get,user, :]
        if conf.no_probs:
            rx_prob = rx_real.unsqueeze(-1).to(DEVICE)
        else:
            rx_prob = torch.cat((rx_real, initial_probs), dim=1).unsqueeze(-1).to(DEVICE)

        if conf.full_e2e:
            train_loss_vect, val_loss_vect = self._train_model(self.detector[0], tx_cur, rx_prob,num_bits, epochs, torch.arange(0, iters_e2e))
        else:
            train_loss_vect, val_loss_vect = self._train_model(self.detector[0], tx_cur, rx_prob,num_bits, epochs, 0)
            probs_vec = self._initialize_probs_for_training(tx, num_bits, n_users)
            # Training the DeepSICe2e for each user-symbol/iteration
            for i in range(1, iters_e2e):
                probs_vec, llrs_mat = self._calculate_posteriors(self.detector, rx_real.to(device=DEVICE).unsqueeze(-1), probs_vec,num_bits, n_users,i)
                # Training the DeepSICe2e networks for the iteration>1
                rx_prob = torch.cat((rx_real.to(device=DEVICE), probs_vec), dim=1).unsqueeze(-1)

                train_loss_cur, val_loss_cur = self._train_model(self.detector[i], tx_cur, rx_prob,num_bits, epochs, 0)
                if SHOW_ALL_ITERATIONS:
                    train_loss_vect = train_loss_vect + train_loss_cur
                    val_loss_vect = val_loss_vect + val_loss_cur

        return train_loss_vect , val_loss_vect

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor) -> torch.Tensor:
        """
        Cross Entropy loss - distribution over states versus the gt state label
        """
        return self.criterion(input=est, target=tx)

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iters_e2e: int, probs_in: torch.Tensor) -> tuple[List, List]:
        # detect and decode
        if conf.full_e2e:
            iters_inference = 1
            detected_word_list = [None]
            llrs_mat_list = [None]
        else:
            iters_inference = iters_e2e
            detected_word_list = [None] * iters_e2e
            llrs_mat_list = [None] * iters_e2e
        probs_vec = self._initialize_probs_for_infer(rx, num_bits, n_users)

        for i in range(iters_inference):
            probs_vec, llrs_mat_list[i] = self._calculate_posteriors(self.detector, rx.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits, n_users, i+1)
            detected_word_list[i] = self._compute_output(probs_vec)

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
                rx_prob_all = rx.unsqueeze(-1)
            else:
                rx_prob_all.append(torch.cat((rx.unsqueeze(-1), probs_vec), dim=1))

            tx_all.append(tx[:, user, :])
        return tx_all, rx_prob_all


    def _initialize_probs_for_training(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        return HALF * torch.ones(dim0,dim1,dim2, dtype=torch.float32).to(DEVICE)

    def _initialize_probs(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        # rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(dim0,dim1,dim2,dim3)).astype(np.float32))
        rnd_init = HALF * torch.ones(dim0,dim1,dim2, dtype=torch.float32)
        return rnd_init

    def _calculate_posteriors(self, model: List[List[nn.Module]], rx_real: torch.Tensor, prob: torch.tensor, num_bits: int, n_users: int, i) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        if conf.no_probs:
            rx_prob = rx_real
        else:
            rx_prob = torch.cat((rx_real, prob.unsqueeze(-1)), dim=1)

        with torch.no_grad():
            if conf.full_e2e:
                output, llrs = model[i-1](rx_prob,num_bits, torch.arange(0, conf.iters_e2e))
            else:
                output, llrs = model[i-1](rx_prob,num_bits, 0)
        probs_mat = output
        llrs_mat = llrs
        return probs_mat, llrs_mat

    def _initialize_probs_for_infer(self, rx: torch.Tensor, num_bits: int, n_users: int):
        dim0 = rx.shape[0]
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        return HALF * torch.ones(dim0,dim1,dim2, dtype=torch.float32).to(DEVICE)
