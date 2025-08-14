import os
from python_code import conf

# Handle GPU/CPU settings
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0  # Use "" for CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError:
    if os.name == 'nt':  # Windows
        os.system("pip install sionna >nul 2>&1")
    else:
        os.system("pip install sionna >/dev/null 2>&1")
    import sionna

import tensorflow as tf
import numpy as np
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder


class LDPC5GCodec:
    def __init__(self, k, n, num_iter=25, hard_out=True):
        """
        k         : Number of information bits
        n         : Number of codeword bits
        num_iter  : Number of BP iterations
        hard_out  : Whether to output hard bits
        """
        self.k = k
        self.n = n
        self.encoder = LDPC5GEncoder(k, n)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_iter, hard_out=hard_out)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        bits: shape [batch_size, k], dtype=bool or int
        Returns encoded codewords: shape [batch_size, n]
        """
        bits_tf = tf.convert_to_tensor(bits, dtype=tf.float32)
        codewords = self.encoder(bits_tf)
        return codewords.numpy()

    def decode(self, llr: np.ndarray) -> np.ndarray:
        """
        llr: shape [batch_size, n], dtype=float (log-likelihood ratios)
        Returns decoded bits: shape [batch_size, k]
        """
        llr_tf = tf.convert_to_tensor(llr, dtype=tf.float32)
        decoded_bits = self.decoder(llr_tf)
        return decoded_bits.numpy()
