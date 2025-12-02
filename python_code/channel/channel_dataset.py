import concurrent.futures
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset

from python_code import DEVICE
from python_code.channel.mimo_channels.mimo_channel_dataset import MIMOChannel
from python_code import conf


class ChannelModelDataset(Dataset):
    """
    Dataset object for the channel. Used in training and evaluation.
    Returns (transmitted, received, channel_coefficients) batch.
    """

    def __init__(self, block_length: int, pilots_length: int, blocks_num: int, num_res: int,
                 clip_percentage_in_tx: int, cfo_and_iqmm_in_rx: bool, kernel_size: int, n_users: int):
        """
        Initialzes the relevant hyperparameters
        :param block_length: number of pilots + data bits
        :param pilots_length: number of pilot bits
        :param blocks_num: number of blocks in the transmission
        it is the block-fading channel used in Section V.B in the original paper.
        """
        self.blocks_num = blocks_num
        if block_length > 0:
            self.block_length = block_length
        else:
            self.block_length = pilots_length*conf.block_length_factor
        self.channel_type = MIMOChannel(self.block_length, pilots_length, clip_percentage_in_tx, cfo_and_iqmm_in_rx, n_users)
        self.num_res = num_res
        self.kernel_size = kernel_size

    def get_snr_data(self, noise_var: float, database: list, num_bits: int, n_users: int, mod_pilot: int, ldpc_k: int, ldpc_n: int):
        if database is None:
            database = []
        tx_full = np.empty((self.blocks_num, self.block_length, self.channel_type.tx_length, self.num_res))
        h_full = np.empty((self.blocks_num, *self.channel_type._h_shape, self.num_res), dtype=np.complex128)
        rx_full = np.empty((self.blocks_num, int(self.block_length / num_bits), self.channel_type.rx_length, self.num_res), dtype=np.complex128)
        rx_ce_full = np.empty((self.blocks_num, n_users, int(self.block_length / num_bits), self.channel_type.rx_length, self.num_res), dtype=np.complex128)
        s_orig_full = np.empty((self.blocks_num, int(self.block_length / num_bits), self.channel_type.tx_length, self.num_res), dtype=np.complex128)
        # accumulate words until reaches desired number
        for index in range(self.blocks_num):
            tx, h, rx, rx_ce, s_orig, rx_clean = self.channel_type._transmit_and_detect(noise_var, self.num_res, index, n_users, mod_pilot, ldpc_k, ldpc_n)
            # accumulate
            tx_full[index] = tx
            rx_full[index] = rx
            rx_ce_full[index] = rx_ce
            h_full[index] = h
            s_orig_full[index] = s_orig

        database.append((tx_full, rx_full, rx_ce_full, h_full, s_orig_full, rx_clean))

    def __getitem__(self, noise_var_list: List[float], num_bits: int, n_users: int, mod_pilot: int, ldpc_k: int, ldpc_n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        database = []
        # do not change max_workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            [executor.submit(self.get_snr_data, noise_var, database, num_bits, n_users, mod_pilot, ldpc_k, ldpc_n) for noise_var in noise_var_list]
        tx, rx, rx_ce, h, s_orig, rx_clean = (np.concatenate(arrays) for arrays in zip(*database))
        tx, rx, rx_ce, h , s_orig, rx_clean = torch.Tensor(tx).to(device=DEVICE), torch.from_numpy(rx).to(device=DEVICE), torch.from_numpy(rx_ce).to(device=DEVICE), torch.from_numpy(
            h).to(device=DEVICE), torch.from_numpy(s_orig).to(device=DEVICE), torch.from_numpy(rx_clean).to(device=DEVICE)
        return tx, rx, rx_ce, h, s_orig, rx_clean

    def __len__(self):
        return self.block_length
