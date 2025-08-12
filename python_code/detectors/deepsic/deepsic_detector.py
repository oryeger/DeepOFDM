import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

HIDDEN_BASE_SIZE = 16

class DeepSICDetector(nn.Module):
    def __init__(self, num_bits, n_users):
        super(DeepSICDetector, self).__init__()
        torch.manual_seed(42)
        if conf.separate_nns:
            conv_num_channels =  int(num_bits/2)*n_users+conf.n_ants*2
        else:
            if conf.half_probs:
                conv_num_channels =  int(num_bits+(num_bits/2)*(n_users-1)+conf.n_ants*2)
            else:
                if conf.train_on_ce_no_pilots or conf.use_data_as_pilots:
                    conv_num_channels = int(num_bits * n_users + conf.n_ants * 4)
                else:
                    if conf.no_probs:
                        conv_num_channels = int(conf.n_ants * 2)
                    else:
                        conv_num_channels = int(num_bits * n_users + conf.n_ants * 2)

        hidden_size = HIDDEN_BASE_SIZE * num_bits
        if conf.scale_input:
            matrix_size = conv_num_channels*conf.num_res # conf.n_ants*2*conf.num_res
            if conf.dot_product:
                self.scale = nn.Parameter(torch.ones(matrix_size))
            else:
                self.fc0 = nn.Linear(matrix_size, matrix_size, bias=False)
                with torch.no_grad():
                    self.fc0.weight.copy_(torch.eye(matrix_size) + 1e-6 * torch.ones(matrix_size, matrix_size))

        self.fc1 = nn.Conv2d(in_channels=conv_num_channels, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1),padding='same')

        if conf.separate_nns:
            self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=int(num_bits/2), kernel_size=(conf.kernel_size, 1),padding='same')
        else:
            # self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1),padding='same')
            self.fc2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(conf.kernel_size, 1), padding='same')

        self.fc3 = nn.Conv2d(in_channels=hidden_size, out_channels=num_bits, kernel_size=(conf.kernel_size, 1), padding='same')

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.Sigmoid()


    def forward(self, rx_prob):
        if conf.scale_input:
            rx = rx_prob # rx = rx_prob[:,:conf.n_ants*2,:,:]
            rx_flattened = rx.reshape(rx.shape[0], rx.shape[1] * rx.shape[2], 1).squeeze(-1)
            if conf.dot_product:
                rx_out_flattened = rx_flattened*self.scale
            else:
                rx_out_flattened = self.fc0(rx_flattened)
            rx_out = rx_out_flattened.unsqueeze(-1).reshape_as(rx)

            out1 = self.activation1(self.fc1(rx_out))
        else:
            out1 = self.activation1(self.fc1(rx_prob))

        out2 = self.activation2(self.fc2(out1))
        llrs = self.fc3(out2)
        out3 = self.activation3(llrs)
        return out3, llrs