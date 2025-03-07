import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sympy import ceiling
from torch.nn import BCELoss
from torch.optim import Adam

from python_code import DEVICE, conf
from python_code.channel.channel_dataset import ChannelModelDataset
from python_code.utils.metrics import calculate_ber
import matplotlib.pyplot as plt
from python_code.utils.constants import (IS_COMPLEX, MOD_PILOT, EPOCHS, NUM_SNRs, NUM_BITS, N_USERS, TRAIN_PERCENTAGE, ITERATIONS, INTERF_FACTOR,
                                         GENIE_CFO, FFT_size, FIRST_CP, CP, NUM_SYMB_PER_SLOT, NUM_SAMPLES_PER_SLOT)

import commpy.modulation as mod

from python_code.channel.modulator import BPSKModulator
import pandas as pd





random.seed(conf.seed)
torch.manual_seed(conf.seed)
torch.cuda.manual_seed(conf.seed)
np.random.seed(conf.seed)

def get_next_divisible(num, divisor):
    return (num + divisor - 1) // divisor * divisor

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
        pilot_size = get_next_divisible(conf.pilot_size,NUM_BITS*NUM_SYMB_PER_SLOT)
        self.pilot_size = pilot_size
        self._initialize_dataloader(num_res,self.pilot_size)
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


    def _initialize_dataloader(self, num_res, pilot_size):
        """
        Sets up the data loader - a generator from which we draw batches, in iterations
        """
        conf.num_res = num_res
        self.channel_dataset = ChannelModelDataset(block_length=conf.block_length,
                                                   pilots_length=pilot_size,
                                                   blocks_num=conf.blocks_num,
                                                   num_res=conf.num_res,
                                                   fading_in_channel=conf.fading_in_channel,
                                                   spatial_in_channel=conf.spatial_in_channel,
                                                   delayspread_in_channel=conf.delayspread_in_channel,
                                                   clip_percentage_in_tx=conf.clip_percentage_in_tx,
                                                   cfo=conf.cfo,
                                                   go_to_td=conf.go_to_td,
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

        if MOD_PILOT == 2:
            mod_text = 'BPSK'
        elif MOD_PILOT == 4:
            mod_text = 'QPSK'
        else:
            mod_text = [str(MOD_PILOT) + 'QAM']
            mod_text = mod_text[0]

        total_ber = []
        total_ber_legacy = []
        total_ber_legacy_genie = []
        SNR_range = [conf.snr + i for i in range(NUM_SNRs)]
        for snr_cur in SNR_range:
            # draw the test words for a given snr

            self._initialize_detector(conf.num_res) # For reseting teh weights

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
                pilot_chunk = int(self.pilot_size / np.log2(MOD_PILOT))
                tx_pilot, tx_data = tx[:self.pilot_size], tx[self.pilot_size:]
                rx_pilot, rx_data = rx_real[:pilot_chunk], rx_real[pilot_chunk:]

                # online training main function
                if self.is_online_training:
                    train_loss_vect , val_loss_vect = self._online_training(tx_pilot, rx_pilot)
                    # Zero CNN weights
                    detected_word, llrs_mat = self._forward(rx_data)

                # CE Based
                # train_loss_vect = [0] * EPOCHS
                # val_loss_vect = [0] * EPOCHS
                rx_data_c = rx[pilot_chunk:].cpu()
                ber_acc = 0
                ber_legacy_acc = 0
                ber_legacy_acc_genie = 0
                for re in range(conf.num_res):
                    H = torch.zeros_like(rx_ce[:, pilot_chunk:, :, re])
                    H_pilot = torch.zeros_like(rx_ce[:, :pilot_chunk, :, re])
                    for user in range(N_USERS):
                        s_orig_data_pilot = s_orig[:pilot_chunk,user,re]
                        rx_data_ce_cur_pilot = rx_ce[user,:pilot_chunk,:,re]
                        H_pilot[user,:,:] = (s_orig_data_pilot[:, None].conj()/(torch.abs(s_orig_data_pilot[:, None]) ** 2)* rx_data_ce_cur_pilot)
                        s_orig_data = s_orig[pilot_chunk:,user,re]
                        rx_data_ce_cur = rx_ce[user,pilot_chunk:,:,re]
                        H[user,:,:] = (s_orig_data[:, None].conj()/(torch.abs(s_orig_data[:, None]) ** 2)* rx_data_ce_cur)
                        pass
                    H_pilot = H_pilot.cpu().numpy()
                    H = H.cpu().numpy()

                    equalized = torch.zeros(rx_data_c.shape[0],tx_data.shape[1], dtype=torch.cfloat)
                    for i in range(rx_data_c.shape[0]):
                        H_cur = H[:,i,:].T
                        H_Ht = H_cur @ H_cur.T.conj()
                        H_Ht_inv = np.linalg.pinv(H_Ht)
                        H_pi = torch.tensor(H_cur.T.conj() @ H_Ht_inv)
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

                    if conf.cfo > 0 & GENIE_CFO:
                        NUM_SLOTS = int(s_orig.shape[0] / NUM_SYMB_PER_SLOT)
                        n = np.arange(int(NUM_SLOTS * NUM_SAMPLES_PER_SLOT))
                        cfo_phase = 2 * np.pi * conf.cfo * n / FFT_size  # CFO phase shift

                        pointer = 0
                        cfo_genie_vect = np.array([])
                        for slot_num in range(NUM_SLOTS):
                            cp_length = FIRST_CP
                            for ofdm_symbol in range(NUM_SYMB_PER_SLOT):
                                pointer += (cp_length + int(FFT_size / 2))
                                cfo_genie_vect = np.concatenate((cfo_genie_vect, np.array([np.exp(1j * cfo_phase[pointer])])))
                                pointer += int(FFT_size / 2)
                                cp_length = CP

                    cfo_genie_vect = cfo_genie_vect[pilot_chunk:]
                    equalized = torch.zeros(rx_data_c.shape[0],tx_data.shape[1], dtype=torch.cfloat)
                    for i in range(rx_data_c.shape[0]):
                        H = h[:, :, re].cpu().numpy()
                        if GENIE_CFO:
                            H = H * cfo_genie_vect[i]
                        H_Ht = H @ H.T.conj()
                        H_Ht_inv = np.linalg.pinv(H_Ht)
                        H_pi = torch.tensor(H.T.conj() @ H_Ht_inv)
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

                    detected_word_cur_re = detected_word[:,:,re,:]
                    detected_word_cur_re = detected_word_cur_re.squeeze(-1)
                    detected_word_cur_re = detected_word_cur_re.reshape(int(tx_data.shape[0]/NUM_BITS), N_USERS, NUM_BITS).swapaxes(1, 2).reshape(tx_data.shape[0], N_USERS)




                    ber = calculate_ber(detected_word_cur_re.cpu(), target.cpu())
                    ber_acc = ber_acc + ber
                    ber_legacy = calculate_ber(detected_word_legacy.cpu(), target.cpu())
                    ber_legacy_acc = ber_legacy_acc + ber_legacy
                    ber_legacy_genie = calculate_ber(detected_word_legacy_genie.cpu(), target.cpu())
                    ber_legacy_acc_genie = ber_legacy_acc_genie + ber_legacy_genie

                    # Just looking at the first user
                    # ber = calculate_ber(detected_word_cur_re[:,0], target[:,0])
                    # ber_acc = ber_acc + ber
                    # ber_legacy = calculate_ber(detected_word_legacy[:,0], target[:,0])
                    # ber_legacy_acc = ber_legacy_acc + ber_legacy
                    # ber_legacy_genie = calculate_ber(detected_word_legacy_genie[:,0], target[:,0])
                    # ber_legacy_acc_genie = ber_legacy_acc_genie + ber_legacy_genie


                total_ber.append(ber)
                total_ber_legacy.append(ber_legacy)
                total_ber_legacy_genie.append(ber_legacy_genie)
                print(f'current DeepSIC:                {block_ind, ber}')
                print(f'current legacy:                 {block_ind, ber_legacy}')
                print(f'current legacy genie:           {block_ind, ber_legacy_genie}')
            # print(f'Final BER: {sum(total_ber) / len(total_ber)}')

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
            epochs_vect = list(range(1, len(train_loss_vect)+1))
            axes[0].plot(epochs_vect[0::conf.num_res], train_loss_vect[0::conf.num_res], linestyle='-', color='b', label='Training Loss')
            axes[0].plot(epochs_vect[0::conf.num_res], val_loss_vect[0::conf.num_res], linestyle='-', color='r', label='Validation Loss')
            axes[0].set_xlabel('Epochs')
            axes[0].set_ylabel('Loss')
            train_samples = int(self.pilot_size*TRAIN_PERCENTAGE/100)
            val_samples =self.pilot_size - train_samples
            title_string = (mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ', SNR=' + str(snr_cur) + ", #REs=" + str(conf.num_res) + ', Interf=' + str(INTERF_FACTOR) + '\n ' +
            'CFO=' + str(conf.cfo) + ' scs, Epochs=' + str(EPOCHS) +  ', #Iterations=' + str(ITERATIONS) + ', CNN kernel size=' + str(conf.kernel_size))

            axes[0].set_title(title_string ,fontsize=10 )
            axes[0].legend()
            axes[0].grid()

            axes[1].hist(llrs_mat.cpu().flatten(), bins=30, color='blue', edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('LLRs')
            axes[1].set_ylabel('#Values')
            axes[1].grid()
            text  = 'BER DeepSIC:' + str(f"{ber:.4f}") + '\
                     BER legacy:' +  str(f"{ber_legacy:.4f}") + '\
                     BER legacy genie:' + (f"{ber_legacy_genie:.4f}")
            # axes[2].text(0.5, 0.5, text, fontsize=12, ha="center", va="center")
            # axes[2].axis('off')
            fig.text(0.15, 0.02, text, ha="left", va="center", fontsize=12)
            plt.show()


            pass



        plt.semilogy(SNR_range, total_ber, '-x', color='b', label='DeeSIC')
        plt.semilogy(SNR_range, total_ber_legacy, '-o', color='r', label='Legacy')
        plt.semilogy(SNR_range, total_ber_legacy_genie, '-o', color='g', label='Legacy Genie')
        plt.xlabel('SNR (dB)')
        plt.ylabel('BER')
        title_string = (mod_text + ', #TRAIN=' + str(train_samples) + ', #VAL=' + str(val_samples) + ", #REs=" + str(conf.num_res) + ', Interf=' + str(INTERF_FACTOR) + '\n ' +
                        'CFO=' + str(conf.cfo) + ' scs, Epochs=' + str(EPOCHS) + ', #Iterations=' + str(ITERATIONS) + ', CNN kernel size=' + str(conf.kernel_size))
        plt.title(title_string, fontsize=10)
        plt.legend()
        plt.grid()
        plt.show()
        df = pd.DataFrame({"SNR_range": SNR_range, "total_ber": total_ber, "total_ber_legacy": total_ber_legacy, "total_ber_legacy_genie": total_ber_legacy_genie})
        print('\n'+title_string)
        title_string = title_string.replace("\n", "")
        df.to_csv("C:\\Projects\\Scatchpad\\" + title_string + ".csv" , index=False)
        # Look at teh weights:
        # print(self.detector[0][0].shared_backbone.fc.weight)
        # print(self.detector[1][0].instance_heads[0].fc1.weight[0])

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
