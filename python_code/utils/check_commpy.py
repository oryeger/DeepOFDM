import numpy as np
import commpy.modulation as mod

qam = mod.QAMModem(4)
const = qam.constellation  # shape (16,)

# Demodulate each constellation point to see the bit label CommPy assigns it
bits_table = qam.demodulate(const, demod_type='hard')  # flat array length 16*4
bits_table = bits_table.reshape(len(const), qam.num_bits_symbol)  # (16,4)

# Optional: print mapping
for i, s in enumerate(const):
    print(i, s, bits_table[i].tolist())
