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
from python_code.utils.constants import IS_COMPLEX, MOD_PILOT, EPOCHS, NUM_SNRs, NUM_BITS, N_USERS, TRAIN_PERCENTAGE, GENIE_CHANNEL

import commpy.modulation as mod

from python_code.channel.modulator import BPSKModulator
import pandas as pd





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

    def __init__(self, num_res):
        # initialize matrices, dataset and detector
        self.lr = 5e-3
        self.is_online_training = True
        self._initialize_dataloader()
        self._initialize_detector(num_res)

    def get_name(self):
        return self.__name__()

    def _initialize_detector(self, num_res):
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
                                                   spatial_in_channel=conf.spatial_in_channel,
                                                   clip_percentage_in_tx=conf.clip_percentage_in_tx,
                                                   kernel_size=conf.kernel_size)

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
        total_ber_legacy_genie = []
        SNR_range = [conf.snr + i for i in range(NUM_SNRs)]
        for snr_cur in SNR_range:
            # draw the test words for a given snr

            self._initialize_detector(12) # For reseting teh weights

            transmitted_words, received_words, received_words_ce, hs, s_orig_words = self.channel_dataset.__getitem__(snr_list=[snr_cur])

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
                tx, h, rx, rx_ce, s_orig = transmitted_words[block_ind], hs[block_ind], received_words[block_ind], received_words_ce[block_ind], s_orig_words[block_ind]
                # Interleave real and imaginary partsos Rx into a real tensor
                if IS_COMPLEX:
                    real_part = rx.real
                    imag_part = rx.imag
                    rx_real = torch.empty((rx.shape[0], rx.shape[1] * 2, rx.shape[2]), dtype=torch.float32)
                    rx_real[:, 0::2, :] = real_part  # Real parts in even rows
                    rx_real[:, 1::2, :] = imag_part  # Imaginary parts in odd rows
                else:
                    rx_real = rx

                # rx_real = rx_real.reshape(rx_real.shape[0]*rx_real.shape[2], rx_real.shape[1])

                # split words into data and pilot part
                pilot_chunk = int(conf.pilot_size / np.log2(MOD_PILOT))
                pilot_chunk_all_res = pilot_chunk*conf.num_res
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
                ber_acc = 0
                ber_legacy_acc = 0
                ber_legacy_acc_genie = 0
                for re in range(conf.num_res):
                    if GENIE_CHANNEL:
                        H = h[:,:,re].numpy()
                    else:
                        H = torch.zeros_like(h[:, :, re])
                        for user in range(N_USERS):
                            s_orig_pilot = s_orig[:pilot_chunk,user,re]
                            rx_pilot_ce_cur = rx_ce[user,:pilot_chunk,:,re]
                            H[:,user] = 1/s_orig_pilot.shape[0]*(s_orig_pilot[:, None].conj()/(torch.abs(s_orig_pilot[:, None]) ** 2)* rx_pilot_ce_cur).sum(dim=0)
                        H = H.numpy()

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

                    # GENIE
                    H = h[:, :, re].numpy()
                    H_Ht = H @ H.T.conj()
                    H_Ht_inv = np.linalg.pinv(H_Ht)
                    H_pi = torch.tensor(H.T.conj() @ H_Ht_inv)
                    equalized = torch.zeros(rx_data_c.shape[0],tx_data.shape[1], dtype=torch.cfloat)
                    for i in range(rx_data_c.shape[0]):
                        equalized[i, :] = torch.matmul(H_pi, rx_data_c[i, :,re])
                    detected_word_legacy_genie = torch.zeros(int(equalized.shape[0]*np.log2(MOD_PILOT)),equalized.shape[1])
                    if MOD_PILOT>2:
                        qam = mod.QAMModem(MOD_PILOT)
                        for i in range(equalized.shape[1]):
                            detected_word_legacy_genie[:,i] = torch.from_numpy(qam.demodulate(equalized[:,i].numpy(),'hard'))
                    else:
                        for i in range(equalized.shape[1]):
                            detected_word_legacy_genie[:,i] = torch.from_numpy(BPSKModulator.demodulate(-torch.sign(equalized[:,i].real).numpy()))


                    # calculate accuracy
                    target = tx_data[:, :rx.shape[1],re]

                    detected_word_cure_re = detected_word[:,:,re,:]
                    detected_word_cure_re = detected_word_cure_re.squeeze(-1)
                    detected_word_cure_re = detected_word_cure_re.reshape(int(tx_data.shape[0]/NUM_BITS), N_USERS, NUM_BITS).swapaxes(1, 2).reshape(tx_data.shape[0], N_USERS)

                    ber = calculate_ber(detected_word_cure_re, target)
                    ber_acc = ber_acc + ber
                    ber_legacy = calculate_ber(detected_word_legacy, target)
                    ber_legacy_acc = ber_legacy_acc + ber_legacy
                    ber_legacy_genie = calculate_ber(detected_word_legacy_genie, target)
                    ber_legacy_acc_genie = ber_legacy_acc_genie + ber_legacy_genie

                total_ber.append(ber)
                total_ber_legacy.append(ber_legacy)
                total_ber_legacy_genie.append(ber_legacy_genie)
                print(f'current DeepSIC:        {block_ind, ber}')
                print(f'current legacy:         {block_ind, ber_legacy}')
                print(f'current legacy genie:   {block_ind, ber_legacy_genie}')
            # print(f'Final BER: {sum(total_ber) / len(total_ber)}')
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
            title_string = 'Loss, ' + mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #EPOCHS=" + str(EPOCHS) + ", SNR=" + str(snr_cur) + ", #REs=" + str(conf.num_res) + ", Clip=" + str(conf.clip_percentage_in_tx) + "%"
            plt.title(title_string ,fontsize=10 )
            plt.legend()
            plt.grid()
            plt.show()

        plt.semilogy(SNR_range, total_ber, '-x', color='b', label='DeeSIC')
        plt.semilogy(SNR_range, total_ber_legacy, '-o', color='r', label='Legacy')
        plt.semilogy(SNR_range, total_ber_legacy_genie, '-o', color='g', label='Legacy Genie')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        title_string = mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #EPOCHS=" + str(EPOCHS) + ", #REs=" + str(conf.num_res) + ", Clip=" + str(conf.clip_percentage_in_tx) + "%"
        plt.title(title_string, fontsize=10)
        plt.legend()
        plt.grid()
        plt.show()
        df = pd.DataFrame({"SNR_range": SNR_range, "total_ber": total_ber, "total_ber_legacy": total_ber_legacy})

        df.to_csv("C:\\Projects\\Scatchpad\\" + title_string + ".csv" , index=False)
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
