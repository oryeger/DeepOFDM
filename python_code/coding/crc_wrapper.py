import os
import tensorflow as tf
import numpy as np

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

from sionna.fec.crc import CRCEncoder, CRCDecoder


class CRC5GCodec:
    def __init__(self, crc_length):
        """
        k           : Number of information bits (before CRC)
        crc_degree  : CRC length in bits
        poly        : CRC polynomial (None = default from Sionna)
        use_decoder : Whether to also create a CRCDecoder
        """
        if crc_length == 24:
            crc_degree="CRC24A"
        else: # crc_length == 16
            crc_degree="CRC16"

        self.encoder = CRCEncoder(crc_degree=crc_degree)
        self.decoder = CRCDecoder(crc_encoder=self.encoder)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """
        bits: shape [batch_size, k], dtype=bool or int
        Returns: shape [batch_size, k+crc_degree]
        """
        bits_tf = tf.convert_to_tensor(bits, dtype=tf.float32)
        encoded = self.encoder(bits_tf)
        return encoded.numpy()

    def decode(self, bits_with_crc: np.ndarray) -> np.ndarray:
        """
        bits_with_crc: shape [batch_size, k+crc_degree]
        Returns: shape [batch_size, k], CRC-checked (invalid frames may be all zeros)
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized. Set use_decoder=True.")
        bits_tf = tf.convert_to_tensor(bits_with_crc, dtype=tf.float32)
        decoded = self.decoder(bits_tf)
        return decoded[1]
