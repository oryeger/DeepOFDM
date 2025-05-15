import torch
from torch import nn
import torch.nn.functional as F

from python_code import conf

from python_code.utils.constants import N_ANTS

import numpy as np
from itertools import product
import commpy.modulation as mod


# 16-QAM constellation points (normalized to unit energy)
qam16 = np.array([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
                  -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                  1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j,
                  3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j]) / np.sqrt(10)


def SphereDecoder(H, y, radius_in):

    """
    Sphere decoder for 4 layers with 16-QAM

    Parameters:
        H: 4x4 complex channel matrix
        y: 4x1 received vector
        radius: initial search radius

    Returns:
        x_hat: estimated transmitted symbol vector
    """

    # generate all constellation points
    num_bits = int(np.log2(conf.mod_pilot))
    bit_combinations = np.array(list(product([0, 1], repeat=num_bits)), dtype=np.int64)
    bit_combinations_flat = bit_combinations.flatten()
    qam = mod.QAMModem(conf.mod_pilot)
    candidates = qam.modulate(bit_combinations_flat)

    # QR decomposition of H
    Q, R = np.linalg.qr(H)
    bits_out = np.zeros((y.shape[0]*num_bits, y.shape[1]))
    for i in range(y.shape[0]):
        radius = radius_in
        y_tilde = Q.conj().T @ y[i,:]

        # Initialize variables
        best_distance_good = float('inf')
        best_s_good = np.ones((conf.n_users),dtype='complex128')*candidates[0]

        # distances_list = []
        # for s3 in candidates:
        #     for s2 in candidates:
        #         for s1 in candidates:
        #             for s0 in candidates:
        #                 current_s = np.array([s0, s1, s2, s3])
        #                 dist = np.sum(np.abs(y[i] -  current_s@H) ** 2)
        #                 distances_list.append(dist)
        #                 if (dist < best_distance_good):
        #                     best_distance_good = dist
        #                     best_s_good = current_s

        # Start with the last layer (layer 3)
        best_distance = float('inf')
        best_s = np.ones((conf.n_users),dtype='complex128')*candidates[0]

        for s3 in candidates:
            dist3 = abs(y_tilde[3] - R[3, 3] * s3) ** 2
            if dist3 > radius:
                continue

            # Layer 2
            for s2 in candidates:
                e2 = y_tilde[2] - R[2, 3] * s3 - R[2, 2] * s2
                dist2 = dist3 + abs(e2) ** 2
                if dist2 > radius:
                    continue

                # Layer 1
                for s1 in candidates:
                    e1 = y_tilde[1] - R[1, 3] * s3 - R[1, 2] * s2 - R[1, 1] * s1
                    dist1 = dist2 + abs(e1) ** 2
                    if dist1 > radius:
                        continue

                    # Layer 0
                    for s0 in candidates:
                        e0 = y_tilde[0] - R[0, 3] * s3 - R[0, 2] * s2 - R[0, 1] * s1 - R[0, 0] * s0
                        total_dist = dist1 + abs(e0) ** 2

                        if total_dist < best_distance:
                            best_distance = total_dist
                            best_s = np.array([s0, s1, s2, s3])
                            radius = best_distance  # Update radius
                            # radius = radius_in
        bits_matrix = np.array([bit_combinations[np.where(candidates == s)[0][0]] for s in best_s]).T
        # bits_matrix = np.array([bit_combinations[np.where(candidates == s)[0][0]] for s in best_s_good]).T
        bits_out[i*num_bits:(i+1)*num_bits,:] = bits_matrix
        # if not(np.array_equal(np.round(np.unique(best_s_good - best_s)), np.array([0. + 0.j]))):
        #     pass
    return bits_out


