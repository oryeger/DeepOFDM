import random
from typing import List, Tuple

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from python_code import DEVICE, conf
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.constants import IS_COMPLEX, MOD_PILOT, EPOCHS, NUM_SNRs, NUM_BITS, N_USERS, TRAIN_PERCENTAGE

import commpy.modulation as mod

from python_code.channel.modulator import BPSKModulator







random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)


class Trainer(object):
    """
    Implements the meta-trainer class. Every trainer must inherent from this base class.
    It implements the evaluation method, initializes the dataloader and the detector.
    It also defines some functions that every inherited trainer must implement.
    """

    def __init__(self):
        # initialize matrices, dataset and detector
        self.lr = 5e-3
        self.is_online_training = True
        self._initialize_dataloader()
        self._initialize_detector()

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self):
        """
        Every trainer must have some base detector
        """
        self.detector = None

    # calculate train loss
    def _calculate_loss(self, est: torch.Tensor, tx: torch.Tensor) -> torch.Tensor:
        """
         Every trainer must have some loss calculation
        """
        pass

    # setup the optimization algorithm
    def _deep_learning_setup(self,single_model):
        """
        Sets up the optimizer and loss criterion
        """
        self.optimizer = Adam(filter(lambda p: p.requires_grad, single_model.parameters()), lr=self.lr)
        self.criterion = BCELoss().to(DEVICE)


    def _initialize_dataloader(self):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        self.channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                   pilots_length=conf.pilot_size,
                                                   blocks_num=conf.blocks_num,
                                                   num_res=conf.num_res,
                                                   fading_in_channel=conf.fading_in_channel,
                                                   spatial_in_channel=conf.spatial_in_channel)

    def _online_training(self, tx: torch.Tensor, rx: torch.Tensor):
        """
        Every detector trainer must have some function to adapt it online
        """
        pass

    def _forward(self, rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Every trainer must have some forward pass for its detector
        """
        pass

    def evaluate(self) -> List[float]:
        """
        The online evaluation run. Main function for running the experiments of sequential transmission of pilots and
        data blocks for the paper.
        :return: list of ber per timestep
        """

        fig_nums = plt.get_fignums()  # Get a list of all figure numbers
        for fig_num in fig_nums[:-1]:  # Exclude the last figure
            plt.close(fig_num)
        plt.close('all')

        total_ber = []
        total_ber_legacy = []
        SNR_range = [conf.snr + i for i in range(NUM_SNRs)]
        for snr_cur in SNR_range:
            # draw the test words for a given snr

            self._initialize_detector() # For reseting teh weights

            transmitted_words, received_words, hs, s_orig_words = self.channel_dataset.__getitem__(snr_list=[snr_cur])

            # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
            # REs = np.arange(conf.num_res)
            # axes[0].plot(REs, np.abs(hs[0,0,0,:]), linestyle='-', color='b', label='Channel')
            # axes[0].set_ylabel('abs(channel)')
            # axes[0].grid()
            # axes[1].plot(REs, np.unwrap(np.angle((hs[0,0,0,:]))), linestyle='-', color='b', label='Channel')
            # axes[1].set_xlabel('REs')
            # axes[1].set_ylabel('angle(channel)')
            # axes[1].grid()
            # axes[0].set_title('Channel with ' + str(conf.num_res) + ' REs')
            # plt.show()

            # detect sequentially
            for block_ind in range(conf.blocks_num):
                print('*' * 20)
                # get current word and channel
                tx, h, rx, s_orig = transmitted_words[block_ind], hs[block_ind], received_words[block_ind], s_orig_words[block_ind]
                # Interleave real and imaginary partsos Rx into a real tensor
                if IS_COMPLEX:
                    real_part = rx.real
                    imag_part = rx.imag
                    rx_real = torch.empty((rx.shape[0], rx.shape[1] * 2, rx.shape[2]), dtype=torch.float32)
                    rx_real[:, 0::2, :] = real_part  # Real parts in even rows
                    rx_real[:, 1::2, :] = imag_part  # Imaginary parts in odd rows
                else:
                    rx_real = rx

                rx_real = rx_real.reshape(rx_real.shape[0],rx_real.shape[1]*conf.num_res)

                # split words into data and pilot part
                pilot_chunk = int(conf.pilot_size / np.log2(MOD_PILOT))
                tx_pilot, tx_data = tx[:conf.pilot_size], tx[conf.pilot_size:]
                rx_pilot, rx_data = rx_real[:pilot_chunk], rx_real[pilot_chunk:]

                # online training main function
                if self.is_online_training:
                    train_loss_vect , val_loss_vect = self._online_training(tx_pilot, rx_pilot)
                    # detect data part after training on the pilot part
                    detected_word = self._forward(rx_data)

                # train_loss_vect = [0] * EPOCHS
                # val_loss_vect = [0] * EPOCHS
                rx_pilot_c, rx_data_c = rx[:pilot_chunk], rx[pilot_chunk:]
                #s_orig_pilot, s_orig_data = s_orig[:pilot_chunk], s_orig[pilot_chunk:]
                #ChanEst = 1/N_ANTS*torch.sum(s_orig_pilot.conj() * rx_pilot_c, dim=1)
                ber_acc = 0
                ber_legacy_acc = 0
                for re in range(conf.num_res):
                    H = h[:,:,re].numpy()
                    H_Ht = H @ H.T.conj()
                    H_Ht_inv = np.linalg.pinv(H_Ht)
                    H_pi = torch.tensor(H.T.conj() @ H_Ht_inv)
                    equalized = torch.zeros(rx_data_c.shape[0],tx_data.shape[1], dtype=torch.cfloat)
                    for i in range(rx_data_c.shape[0]):
                        equalized[i, :] = torch.matmul(H_pi, rx_data_c[i, :,re])
                    detected_word_legacy = torch.zeros(int(equalized.shape[0]*np.log2(MOD_PILOT)),equalized.shape[1])
                    if MOD_PILOT>2:
                        qam = mod.QAMModem(MOD_PILOT)
                        for i in range(equalized.shape[1]):
                            detected_word_legacy[:,i] = torch.from_numpy(qam.demodulate(equalized[:,i].numpy(),'hard'))
                    else:
                        for i in range(equalized.shape[1]):
                            detected_word_legacy[:,i] = torch.from_numpy(BPSKModulator.demodulate(-torch.sign(equalized[:,i].real).numpy()))
                    pass



                    # calculate accuracy
                    target = tx_data[:, :rx.shape[1],re]

                    indexes = []
                    for user in range(N_USERS):
                        indexes.append(list(range(user * conf.num_res * NUM_BITS + re*NUM_BITS, NUM_BITS * (user * conf.num_res + 1)+re*NUM_BITS)))
                    indexes = sum(indexes, [])

                    ber = calculate_ber(detected_word[:,indexes], target)
                    ber_acc = ber_acc + ber
                    ber_legacy = calculate_ber(detected_word_legacy, target)
                    ber_legacy_acc = ber_legacy_acc + ber_legacy

                total_ber.append(ber)
                total_ber_legacy.append(ber_legacy)
                print(f'current: {block_ind, ber}')
            print(f'Final BER: {sum(total_ber) / len(total_ber)}')
            epochs_vect = list(range(1, len(train_loss_vect)+1))
            plt.plot(epochs_vect, train_loss_vect, linestyle='-', color='b', label='Training Loss')
            plt.plot(epochs_vect, val_loss_vect, linestyle='-', color='r', label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            if MOD_PILOT == 2:
                mod_text = 'BPSK'
            elif MOD_PILOT == 4:
                mod_text = 'QPSK'
            else:
                mod_text = [str(MOD_PILOT)+'QAM']
                mod_text = mod_text[0]


            train_samples = int(conf.pilot_size*TRAIN_PERCENTAGE/100)
            val_samples =conf. pilot_size - train_samples

            plt.title('Loss, ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #EPOCHS=" + str(EPOCHS) + ", SNR=" + str(snr_cur) )
            plt.legend()
            plt.grid()
            plt.show()

        plt.semilogy(SNR_range, total_ber, '-x', color='b', label='DeeSIC')
        plt.semilogy(SNR_range, total_ber_legacy, '-o', color='r', label='Legacy')

        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        plt.title('Total BER, ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #EPOCHS=" + str(EPOCHS))
        plt.legend()
        plt.grid()
        plt.show()
        return total_ber

    def run_train_loop(self, est: torch.Tensor, tx: torch.Tensor) -> float:
        # calculate loss
        loss = self._calculate_loss(est=est, tx=tx)
        current_loss = loss.item()
        # back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return current_loss
