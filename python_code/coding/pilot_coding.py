"""CRC+LDPC encoding of the pilot region (encode_pilots: True).

Kept separate from mimo_channel_dataset so it can be unit-tested without
pulling in the channel-model import chain (sionna.channel, commpy, ...).
"""
import numpy as np


def encode_pilots(bits_generator, pilot_length: int, num_res: int, n_users: int,
                  codec, crc, ldpc_k: int, ldpc_n: int) -> np.ndarray:
    """CRC+LDPC-encode the pilot region slot-wise, same stream layout as the
    data region ((bit_row, re) row-major per user), so code-structure losses
    (training_loss='tsyn') see real codewords on the pilot LLRs. Any remainder
    bits that don't fill a whole slot stay random."""
    total_bits = pilot_length * num_res
    num_slots = int(np.floor(total_bits / ldpc_n))
    coded = bits_generator.integers(0, 2, size=(n_users, total_bits)).astype(float)
    uncoded = bits_generator.integers(0, 2, size=(n_users, num_slots * ldpc_k))
    for slot in range(num_slots):
        with_crc = crc.encode(uncoded[:, slot * ldpc_k:(slot + 1) * ldpc_k])
        coded[:, slot * ldpc_n:(slot + 1) * ldpc_n] = codec.encode(with_crc)
    return coded.reshape(n_users, pilot_length, num_res).transpose(1, 0, 2).astype(int)
