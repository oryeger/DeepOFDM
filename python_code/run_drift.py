#!/usr/bin/env python
"""
run_drift.py — Sequential CFO / channel drift experiment.

Trains with BCE at CFO=0 (and/or channel_drift_index=0), then adapts using a
blind loss as CFO and/or the TDL channel (via channel_drift_index) drift
gradually. Each SNR chains weights from the same SNR in the previous step.

Usage:
    python run_drift.py --config path/to/config.yaml

Edit DRIFT_STEPS below to define the sequence.
"""

import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_code import conf
from python_code.evaluate import run_evaluate, get_mcs
from python_code.detectors.escnn.escnn_trainer import ESCNNTrainer
from python_code.detectors.deepsice2e.deepsice2e_trainer import DeepSICe2eTrainer
from python_code.detectors.deeprx.deeprx_trainer import DeepRxTrainer
from python_code.detectors.deepsic.deepsic_trainer import DeepSICTrainer
from python_code.detectors.deepsicmb.deepsicmb_trainer import DeepSICMBTrainer
from python_code.detectors.deepstag.deepstag_trainer import DeepSTAGTrainer
from python_code.detectors.mhsa.mhsa_trainer import MHSATrainer
from python_code.detectors.tdcnn.tdcnn_trainer import TDCNNTrainer
from python_code.detectors.tdfdcnn.tdfdcnn_trainer import TDFDCNNTrainer
from python_code.detectors.jointllr.jointllr_trainer import JointLLRTrainer

# ── Drift sequence ────────────────────────────────────────────────────────────
# Each step: CFO to train/test at, loss to use, which layers to freeze, and
# channel_drift_index (TDL channel slot-offset; 0 = ordinary/no channel
# drift). First step should use 'bce' (supervised); subsequent steps use
# 'gfmi'/'tent'/'tsyn' (blind). channel_drift_index is independent of cfo -
# set both to drift CFO and the channel together, or leave
# channel_drift_index at 0 for CFO-only drift as before.
DRIFT_STEPS = [
    dict(cfo=0.0, training_loss='bce',  escnn_load_freeze='none', channel_drift_index=0),
    dict(cfo=0.1, training_loss='gfmi', escnn_load_freeze='none', channel_drift_index=0),
    dict(cfo=0.2, training_loss='gfmi', escnn_load_freeze='none', channel_drift_index=0),
    dict(cfo=0.3, training_loss='gfmi', escnn_load_freeze='none', channel_drift_index=0),
    dict(cfo=0.4, training_loss='gfmi', escnn_load_freeze='none', channel_drift_index=0),
]
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Sequential CFO drift experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to base config YAML (all non-drift parameters)')
    args = parser.parse_args()

    conf.reload_config(args.config)

    num_bits_data, _ = get_mcs(conf.mcs)
    num_bits_data = int(num_bits_data)
    num_bits_pilot = int(np.log2(conf.mod_pilot)) if conf.mod_pilot > 0 else num_bits_data

    # Create trainers once — they read conf dynamically so picking up per-step patches
    escnn_trainer      = ESCNNTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deepsice2e_trainer = DeepSICe2eTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deeprx_trainer     = DeepRxTrainer(conf.num_res, conf.n_users, conf.n_ants)
    deepsic_trainer    = DeepSICTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deepsicmb_trainer  = DeepSICMBTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    deepstag_trainer   = DeepSTAGTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    mhsa_trainer       = MHSATrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    tdcnn_trainer      = TDCNNTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    tdfdcnn_trainer    = TDFDCNNTrainer(num_bits_pilot, conf.n_users, conf.n_ants)
    jointllr_trainer   = JointLLRTrainer(num_bits_pilot, conf.n_users, conf.n_ants)

    prev_tag = None
    total_start = time.time()

    for step_idx, step in enumerate(DRIFT_STEPS):
        print(f"\n{'='*70}", flush=True)
        channel_drift_index = step.get('channel_drift_index', 0)
        print(f"[DRIFT] Step {step_idx + 1}/{len(DRIFT_STEPS)}: "
              f"cfo={step['cfo']}  loss={step['training_loss']}  "
              f"freeze={step['escnn_load_freeze']}  "
              f"channel_drift_index={channel_drift_index}  "
              f"load_tag={prev_tag or '(none — train from scratch)'}", flush=True)
        print(f"{'='*70}\n", flush=True)

        conf.set_value('cfo',                  step['cfo'])
        conf.set_value('training_loss',        step['training_loss'])
        conf.set_value('escnn_load_freeze',    step['escnn_load_freeze'])
        conf.set_value('channel_drift_index',  channel_drift_index)
        conf.set_value('save_escnn_weights',   True)
        conf.set_value('load_escnn_weights_tag', prev_tag or '')

        _, weights_tag = run_evaluate(
            escnn_trainer, deepsice2e_trainer, deeprx_trainer,
            deepsic_trainer, deepsicmb_trainer, deepstag_trainer,
            mhsa_trainer, tdcnn_trainer, tdfdcnn_trainer, jointllr_trainer,
        )

        prev_tag = weights_tag
        print(f"\n[DRIFT] Step {step_idx + 1} complete. saved tag={weights_tag}", flush=True)

    elapsed = time.time() - total_start
    print(f"\n[DRIFT] All {len(DRIFT_STEPS)} steps complete in "
          f"{elapsed / 3600:.1f}h ({elapsed:.0f}s).", flush=True)


if __name__ == '__main__':
    main()
