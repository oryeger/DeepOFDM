
import math
from datetime import datetime
from typing import List

import torch
from torch import nn
from torch.func import functional_call
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.coding.ekf_tracker import EkfParamTracker
from python_code.coding.mcs_table import get_mcs
from python_code.detectors.escnn.escnn_detector import ESCNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE, NUM_SYMB_PER_SLOT
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable
from python_code.utils.probs_utils import pilot_third_bit_mask
from python_code.utils.probs_utils import relevant_indices

Softmax = torch.nn.Softmax(dim=1)

# (beta1, beta2) for ekf.py's persistent per-network Adam (see _deep_learning_setup).
_PERSIST_ADAM_BETAS = (0.0, 0.99)


class ESCNNTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'ESCNN'

    def _deep_learning_setup(self, single_model):
        """
        Builds self.optimizer/self.criterion for one _train_model call. By default a fresh Adam
        every call - evaluate.py always gets this. With self.persist_optimizer set (only ever by
        ekf.py, for its sgd_* modes), each network instead keeps one Adam for the whole run, so
        its moment estimates carry across groups: a fresh Adam's first step is lr*sign(grad)
        on every weight regardless of gradient size, which is all a per-group epochs=1 update
        ever got. The cached Adam is rebuilt if that network's trainable-parameter set changed
        (set_stage/set_load_freeze), since it'd otherwise be stepping the wrong tensors.
        The persistent Adam uses _PERSIST_ADAM_BETAS, not torch's (0.9, 0.999): with one
        step per group, beta1=0.9 momentum keeps pushing along ~10 groups' worth of stale
        gradient after the model has already passed the optimum, and beta2=0.999 lets the
        large gradients of that overshoot hold the step size down for ~1000 groups.
        """
        weight_decay = float(getattr(conf, 'escnn_weight_decay', 0.0))
        trainable = [p for p in single_model.parameters() if p.requires_grad]
        if getattr(self, 'persist_optimizer', False):
            cache = self.__dict__.setdefault('_optimizer_cache', {})
            key = id(single_model)
            param_ids = tuple(id(p) for p in trainable)
            cached = cache.get(key)
            if cached is None or cached[0] != param_ids:
                cached = (param_ids, Adam(trainable, lr=self.lr, betas=_PERSIST_ADAM_BETAS,
                                          weight_decay=weight_decay))
                cache[key] = cached
            self.optimizer = cached[1]
        else:
            self.optimizer = Adam(trainable, lr=self.lr, weight_decay=weight_decay)
        from torch.nn import BCEWithLogitsLoss
        self.criterion = BCEWithLogitsLoss(reduction='none').to(DEVICE)

    def _initialize_detector(self, num_bits, n_users, n_ants):

        self.detector = [[ESCNNDetector(num_bits, n_users).to(DEVICE) for _ in range(conf.iterations)] for _ in
                         range(n_users)]  # 2D list for Storing the ESCNN Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx_prob: torch.Tensor, num_bits:int, epochs: int, first_half_flag: bool, stage: str, network_id: str = "",
                      payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT, real_bit_idx=None) -> list[float]:
        """
        Trains a ESCNN Network and returns the total training loss.

        real_bit_idx: when given, an iterable of bit-channel indices (0..num_bits-1) - only
        these are included in the loss, uniformly across every symbol and RE (unlike the
        make_64QAM_16QAM_percentage mask below, which varies by symbol but never restricts by
        RE). ekf.py's weights_track_mode='sgdbce' passes relevant_indices(qm, qm/2) here: DMRS
        is always QPSK regardless of qm (module docstring), so only 2 of a wider network's qm
        bit channels ever have a real label - see _build_dmrs_bce_training_data. None (default,
        every other caller) means no extra restriction.

        payload_symbols_per_slot: unit (in rx_prob rows) tsyn's slot-alignment/codeword-window
        logic treats as one slot - defaults to a full NUM_SYMB_PER_SLOT (evaluate.py's own pilot
        region). ekf.py's callers pass their own DMRS-stripped payload-only count instead (see
        _get_syndrome_helper's docstring), and a non-default value here doubles as the signal
        that this is one of ekf.py's streaming calls, not evaluate.py's: it always switches off
        escnn_use_primary_val_only's first-half cut (ekf.py's DeepSIC/DeepRx are always frozen
        pretrained checkpoints - see its module docstring - so there's no separate
        primary-detector training pass within this run to leak from, unlike evaluate.py).
        evaluate.py never passes this argument, so that stays exactly as conf says for every one
        of its calls.

        Separately, for ekf.py's weights_track_mode in ('sgdsyn', 'sgdbce') - its two
        calib-region-free modes, which train directly on the group's own scored data or DMRS
        pilots respectively rather than a separate calib region (see ekf.py's module docstring)
        - the train/val split (TRAIN_PERCENTAGE) is skipped entirely too: there's no held-out
        portion to validate against when training runs on exactly what gets scored right
        afterward. ekf.py's weights_track_mode='sgdbcei' (a separate calib region) keeps the
        normal split - that's a real supervised fit with genuine unseen data to validate
        against. sgdsyn is identified via conf.training_loss=='tsyn' (the only mode that sets
        it); sgdbce sets conf.training_loss='bce' too (same as sgdbcei), so it's identified by
        reading conf.weights_track_mode directly instead.
        """
        single_model = single_model.to(DEVICE)

        if isinstance(single_model, ESCNNDetector):
            single_model.set_stage(stage)

        self._deep_learning_setup(single_model)
        train_loss_vect = []
        val_loss_vect = []

        # Reshape tx to match the expected format
        tx_reshaped = tx.reshape(int(tx.shape[0] // num_bits), num_bits, tx.shape[1])

        # Mask out bits forced constant by the QPSK/16QAM thirds of the 64QAM pilot
        # mix (make_64QAM_16QAM_percentage == 50), so the loss only sees real bits.
        loss_mask = torch.ones_like(tx_reshaped, dtype=torch.bool)
        if conf.make_64QAM_16QAM_percentage == 50 and num_bits == 6:
            bit_mask = pilot_third_bit_mask(tx_reshaped.shape[0], num_bits)
            loss_mask = bit_mask.unsqueeze(-1).expand_as(tx_reshaped)

        # Uniform bit-channel restriction (e.g. ekf.py's sgdbce: only 2 of qm bits are ever
        # real for QPSK-only DMRS) - ANDed with whatever the 64QAM-thirds mask above already set.
        if real_bit_idx is not None:
            real_bit_mask = torch.zeros(num_bits, dtype=torch.bool)
            real_bit_mask[list(real_bit_idx)] = True
            loss_mask = loss_mask & real_bit_mask.view(1, num_bits, 1)

        # tsyn: codeword windows start at symbol 0, so every region the loss
        # sees must begin on a slot boundary (payload_symbols_per_slot rows).
        _slot_align = (getattr(conf, 'training_loss', 'bce') == 'tsyn')

        # A non-default payload_symbols_per_slot only ever comes from ekf.py's DMRS-stripped
        # streaming calls (evaluate.py always leaves it at NUM_SYMB_PER_SLOT) - see this
        # method's docstring for why that means: no primary-detector cut always, and (only for
        # ekf.py's weights_track_mode in ('sgdsyn', 'sgdbce') - its two calib-region-free
        # modes, trained directly on the group's own scored data/DMRS pilots respectively) no
        # val split either. Both sgdsyn and sgdbce set conf.training_loss to 'tsyn'/'bce'
        # respectively (see ekf.py's main()), so training_loss alone can't tell sgdbce apart
        # from sgdbcei (both 'bce') - hence reading weights_track_mode directly here too.
        _ekf_style = (payload_symbols_per_slot != NUM_SYMB_PER_SLOT)
        # sgdht's BCE calls also train on the group's own scored slots (CRC-verified labels), so no split.
        _no_val_split = _ekf_style and (_slot_align or getattr(conf, 'weights_track_mode', 'ekf') in ('sgdbce', 'sgdht'))

        # Restrict to primary detector's validation portion only
        _primary_val_only = False if _ekf_style else getattr(conf, 'escnn_use_primary_val_only', False)
        if _primary_val_only:
            primary_train_samples = rx_prob.shape[0] // 2
            if _slot_align:
                primary_train_samples -= primary_train_samples % payload_symbols_per_slot
            rx_prob = rx_prob[primary_train_samples:]
            tx_reshaped = tx_reshaped[primary_train_samples:]
            loss_mask = loss_mask[primary_train_samples:]

        # Shuffle samples before train/val split to decorrelate from augmenter's own split
        if getattr(conf, 'shuffle_augment_priors', False):
            aug_seed = getattr(conf, 'shuffle_augment_seed', -1)
            generator = torch.Generator().manual_seed(aug_seed) if aug_seed >= 0 else None
            perm = torch.randperm(rx_prob.shape[0], generator=generator)
            rx_prob = rx_prob[perm]
            tx_reshaped = tx_reshaped[perm]
            loss_mask = loss_mask[perm]

        # Split into train and validation sets - none at all for ekf.py's tsyn data-direct calls
        # (training directly on a group's own scored data has nothing to hold out - see this
        # method's docstring); ekf.py's calib-region calls (any other loss) keep the normal split.
        train_samples = rx_prob.shape[0] if _no_val_split else int(rx_prob.shape[0] * TRAIN_PERCENTAGE / 100)
        if _slot_align:
            train_samples -= train_samples % payload_symbols_per_slot
        rx_prob_train = rx_prob[:train_samples]
        rx_prob_val = rx_prob[train_samples:]
        tx_train = tx_reshaped[:train_samples]
        tx_val = tx_reshaped[train_samples:]
        mask_train = loss_mask[:train_samples]
        mask_val = loss_mask[train_samples:]
        # No validation split: there's nothing to early-stop or checkpoint-select against, so
        # disable that machinery below rather than let it operate on an empty tensor (an empty
        # batch's loss reduces to 0.0, which would look like an instant "best" checkpoint at
        # epoch 1 and then never improve again).
        has_val = rx_prob_val.shape[0] > 0

        # Create DataLoader for mini-batch training
        # batch_size <= 0 means full-batch (no mini-batching)
        batch_size = conf.batch_size if hasattr(conf, 'batch_size') else 32
        if batch_size <= 0:
            batch_size = len(rx_prob_train)  # Full batch
        shuffle = conf.shuffle if hasattr(conf, 'shuffle') else True
        train_dataset = TensorDataset(rx_prob_train, tx_train, mask_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        es_patience = int(getattr(conf, 'early_stopping_patience', -1))
        es_enabled = (es_patience > 0) and has_val
        best_val_loss = float('inf')
        epochs_since_best = 0
        best_state = None

        log_every = int(getattr(conf, 'log_train_every_epochs', 0))
        log_enabled = log_every > 0
        tag = f"[ESCNN {network_id}]" if network_id else "[ESCNN]"
        if log_enabled:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag} training start (train={len(rx_prob_train)} val={len(rx_prob_val)} epochs={epochs} stage={stage})", flush=True)

        stopped_early = False
        last_epoch = epochs
        _log_hb = (getattr(conf, 'training_loss', 'bce') in ('gfmi', 'tent', 'tsyn')
                   and getattr(conf, 'beta_balance', 0.0) > 0.0)
        _log_tsyn = getattr(conf, 'training_loss', 'bce') == 'tsyn'
        # Collapse diagnostics (true_ber/const_pos/frac_ones) need ground-truth tx
        # but nothing syndrome-specific, so they're equally meaningful for 'tent'
        # (also blind/unsupervised, same collapse failure mode) as for 'tsyn'.
        _log_collapse = getattr(conf, 'training_loss', 'bce') in ('tent', 'tsyn')

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_hb = 0.0
            epoch_tent = 0.0
            epoch_synd = 0.0
            epoch_sat = 0.0
            num_synd_batches = 0
            num_batches = 0

            # Mini-batch training
            for batch_rx, batch_tx, batch_mask in train_loader:
                batch_rx = batch_rx.to(DEVICE)
                batch_tx = batch_tx.to(DEVICE)
                batch_mask = batch_mask.to(DEVICE)

                soft_estimation, llrs = single_model(batch_rx)

                if first_half_flag:
                    llrs_cur = llrs[:, 0::2, :, :]
                    batch_tx_cur = batch_tx[:, 0::2, :]
                    batch_mask_cur = batch_mask[:, 0::2, :]
                else:
                    llrs_cur = llrs
                    batch_tx_cur = batch_tx
                    batch_mask_cur = batch_mask

                loss = self._calculate_loss(llrs_cur, batch_tx_cur, batch_mask_cur,
                                             payload_symbols_per_slot=payload_symbols_per_slot)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if _log_hb:
                    epoch_hb += self._calc_h_marginal(llrs_cur, batch_mask_cur)
                if _log_tsyn:
                    st = getattr(self, '_tsyn_stats', None)
                    if st is not None:
                        epoch_tent += st['l_tent']
                        if st['l_synd'] is not None:
                            epoch_synd += st['l_synd']
                            epoch_sat += st['sat']
                            num_synd_batches += 1
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_train_hb = epoch_hb / num_batches if num_batches > 0 else 0.0
            train_loss_vect.append(avg_train_loss)

            # Calculate validation loss (skipped entirely when there's no held-out split - see
            # has_val above)
            if has_val:
                with torch.no_grad():
                    rx_prob_val_device = rx_prob_val.to(DEVICE)
                    _, llrs_val = single_model(rx_prob_val_device)
                    if first_half_flag:
                        llrs_val_cur = llrs_val[:, 0::2, :, :]
                        mask_val_cur = mask_val[:, 0::2, :].to(DEVICE)
                        val_loss = self._calculate_loss(llrs_val_cur, tx_val[:, 0::2, :].to(DEVICE), mask_val_cur,
                                                         payload_symbols_per_slot=payload_symbols_per_slot)
                    else:
                        llrs_val_cur = llrs_val
                        mask_val_cur = mask_val.to(DEVICE)
                        val_loss = self._calculate_loss(llrs_val_cur, tx_val.to(DEVICE), mask_val_cur,
                                                         payload_symbols_per_slot=payload_symbols_per_slot)
                    val_loss_vect.append(val_loss.item())
                    val_hb = self._calc_h_marginal(llrs_val_cur, mask_val_cur) if _log_hb else None
                    val_tsyn = dict(getattr(self, '_tsyn_stats', {})) if _log_tsyn else None

                    # ---- TEMP DEBUG: tsyn collapse investigation (remove once done) ----
                    # Ground-truth diagnostics: is the network collapsing to an
                    # input-independent constant output, a sign-biased output, or a
                    # non-trivial-but-fixed syndrome-satisfying codeword? Read-only,
                    # does not feed back into loss/gradients.
                    if _log_collapse:
                        tx_val_cur = (tx_val[:, 0::2, :] if first_half_flag else tx_val).to(DEVICE)
                        # Project convention (see syndrome_loss.py docstring / _compute_output):
                        # sigmoid(L) = P(bit=1), so L>0 => bit 1, matching prob_to_BPSK_symbol's
                        # p>0.5 => bit 1 threshold.
                        hard_val = (llrs_val_cur.squeeze(-1) > 0).long()
                        true_val = tx_val_cur.long()
                        mbool = mask_val_cur.bool()
                        if mbool.any():
                            true_ber = (hard_val[mbool] != true_val[mbool]).float().mean().item()
                            frac_ones = hard_val[mbool].float().mean().item()
                        else:
                            true_ber = float('nan')
                            frac_ones = float('nan')
                        # Per bit-position variance across the batch dim: near-zero
                        # means the network outputs that position the same way
                        # regardless of input (input-independent collapse).
                        var_per_pos = hard_val.float().var(dim=0, unbiased=False)
                        const_pos = (var_per_pos < 1e-6).float().mean().item()
                        _tsyn_debug_str = (f" true_ber={true_ber:.4f} const_pos={const_pos:.4f}"
                                           f" frac_ones={frac_ones:.4f}")
                    else:
                        _tsyn_debug_str = ""
                    # ---- END TEMP DEBUG ----
            else:
                val_loss = None
                val_hb = None
                val_tsyn = None
                _tsyn_debug_str = ""

            def _val_str():
                return f"{val_loss.item():.4f}" if has_val else "n/a"

            def _hb_suffix(train_hb, v_hb):
                if v_hb is None:
                    return f" Hb(q̄): train={train_hb:.4f}"
                return f" Hb(q̄): train={train_hb:.4f} val={v_hb:.4f}"

            def _tsyn_suffix():
                # Raw components logged separately: their scales are not
                # comparable (grid entropy vs check log-penalty) — needed for tw tuning.
                s = f" tent={epoch_tent / max(num_batches, 1):.4f}"
                if num_synd_batches > 0:
                    s += (f" synd={epoch_synd / num_synd_batches:.4f}"
                          f" sat={epoch_sat / num_synd_batches:.3f}")
                if val_tsyn and val_tsyn.get('l_synd') is not None:
                    s += (f" val_synd={val_tsyn['l_synd']:.4f}"
                          f" val_sat={val_tsyn['sat']:.3f}")
                return s

            if es_enabled:
                cur_val = val_loss.item()
                if cur_val < best_val_loss:
                    best_val_loss = cur_val
                    epochs_since_best = 0
                    best_state = {k: v.detach().clone() for k, v in single_model.state_dict().items()}
                else:
                    epochs_since_best += 1
                    if epochs_since_best > es_patience:
                        stopped_early = True
                        last_epoch = epoch + 1
                        if log_enabled:
                            msg = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} epoch {epoch+1}/{epochs} train={avg_train_loss:.4f} val={val_loss.item():.4f} best_val={best_val_loss:.4f}"
                            if _log_hb:
                                msg += _hb_suffix(avg_train_hb, val_hb)
                            if _log_tsyn:
                                msg += _tsyn_suffix()
                            if _log_collapse:
                                msg += _tsyn_debug_str  # TEMP DEBUG: tsyn collapse investigation
                            print(msg, flush=True)
                        break

            if log_enabled and (epoch + 1) % log_every == 0:
                msg = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} epoch {epoch+1}/{epochs} train={avg_train_loss:.4f} val={_val_str()}"
                if es_enabled:
                    msg += f" best_val={best_val_loss:.4f}"
                if _log_hb:
                    msg += _hb_suffix(avg_train_hb, val_hb)
                if _log_tsyn:
                    msg += _tsyn_suffix()
                if _log_collapse:
                    msg += _tsyn_debug_str  # TEMP DEBUG: tsyn collapse investigation
                print(msg, flush=True)

        if not stopped_early:
            last_epoch = epochs

        if es_enabled and best_state is not None:
            single_model.load_state_dict(best_state)

        if log_enabled:
            reason = "early stop" if stopped_early else "completed"
            final_val = best_val_loss if es_enabled else (val_loss_vect[-1] if val_loss_vect else float('nan'))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {tag} done @ epoch {last_epoch} ({reason}) val={final_val:.4f}", flush=True)

        return train_loss_vect, val_loss_vect

    def _train_models(self, model: List[List[ESCNNDetector]], i: int, tx_all: List[torch.Tensor],
                      rx_prob_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int, first_half_flag: bool, stage: str,
                      payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT, real_bit_idx=None):
        """Returns (train_loss_vect_users, val_loss_vect_users): one loss-per-epoch
        list per UE (each UE has its own network, so its own loss history)."""
        train_loss_vect_users = [None] * n_users
        val_loss_vect_users = [None] * n_users
        for user in range(n_users):
            net_id = f"u={user} it={i}"
            train_loss_vect , val_loss_vect = self._train_model(model[user][i], tx_all[user], rx_prob_all[user].to(DEVICE), num_bits, epochs, first_half_flag, stage, network_id=net_id,
                                                                  payload_symbols_per_slot=payload_symbols_per_slot,
                                                                  real_bit_idx=real_bit_idx)
            train_loss_vect_users[user] = train_loss_vect
            val_loss_vect_users[user] = val_loss_vect
        return train_loss_vect_users , val_loss_vect_users




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor, stage: str = "base",
                          payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT, real_bit_idx=None):
        """
        Main training function for ESCNN trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).

        payload_symbols_per_slot/real_bit_idx are passed straight through to _train_model (see
        its docstring) - defaults reproduce evaluate.py's existing behavior untouched; ekf.py's
        callers override them explicitly.
        """

        if conf.which_augment == 'NO_AUGMENT':
            initial_probs = self._initialize_probs(tx, num_bits, n_users)
        else:
            initial_probs = probs_in

        # Training the ESCNN network for each user for iteration=1
        tx_all, rx_prob_all = self._prepare_data_for_training(tx, rx_real, initial_probs, n_users)
        # ---- TEMP DEBUG: overlapping per-UE loss curves investigation (remove once done) ----
        _labels_identical = [bool(torch.equal(tx_all[0], tx_all[u])) for u in range(1, n_users)]
        print(f"[TEMP DEBUG] tx_all[u] identical to tx_all[0] for u=1..{n_users-1}: {_labels_identical}", flush=True)
        # ---- END TEMP DEBUG ----
        train_loss_vect , val_loss_vect = self._train_models(self.detector, 0, tx_all, rx_prob_all, num_bits, n_users, epochs, first_half_flag, stage,
                                                              payload_symbols_per_slot=payload_symbols_per_slot,
                                                              real_bit_idx=real_bit_idx)
        # ---- TEMP DEBUG: overlapping per-UE loss curves investigation (remove once done) ----
        _final_losses = [round(v[-1], 8) if v else None for v in train_loss_vect]
        _vect_identical = [train_loss_vect[u] == train_loss_vect[0] for u in range(1, n_users)]
        print(f"[TEMP DEBUG] final train loss per UE: {_final_losses}", flush=True)
        print(f"[TEMP DEBUG] train_loss_vect[u] == train_loss_vect[0] for u=1..{n_users-1}: {_vect_identical}", flush=True)
        # ---- END TEMP DEBUG ----
        # Initializing the probabilities
        if conf.which_augment == 'NO_AUGMENT':
            probs_vec = self._initialize_probs_for_training(tx, num_bits, n_users)
        else:
            probs_vec = probs_in.to(DEVICE)
        # Training the ESCNN for each user-symbol/iteration
        for i in range(1, iterations):
            # Training the ESCNN networks for the iteration>1
            # Generating soft symbols for training purposes
            probs_vec, llrs_mat = self._calculate_posteriors(self.detector, i, rx_real.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits,n_users, 0)
            tx_all, rx_prob_all = self._prepare_data_for_training(tx, rx_real.to(device=DEVICE), probs_vec, n_users)
            train_loss_cur , val_loss_cur =  self._train_models(self.detector, i, tx_all, rx_prob_all, num_bits, n_users, epochs, first_half_flag, stage,
                                                                 payload_symbols_per_slot=payload_symbols_per_slot,
                                                                 real_bit_idx=real_bit_idx)
            if SHOW_ALL_ITERATIONS:
                for user in range(n_users):
                    train_loss_vect[user] = train_loss_vect[user] + train_loss_cur[user]
                    val_loss_vect[user] = val_loss_vect[user] + val_loss_cur[user]
        return train_loss_vect , val_loss_vect

    def _get_ekf_trackers(self):
        """Lazily build one EkfParamTracker per (user, iteration) network; persists on this
        trainer instance across blocks *and* SNR steps, since the trainer itself is created
        once per run. evaluate.py rebuilds self.detector every SNR
        via _initialize_detector+load_weights, so existing trackers get rebind()'d onto the
        new module objects rather than recreated from scratch - that (plus rebind()'s no-op
        on an already-bound net) is what carries the online-tracked (theta, P) state across
        SNR steps while still refreshing theta_pretrained to each SNR's own checkpoint."""
        dynamics = getattr(conf, 'escnn_ekf_dynamics', 'ar1')
        alpha = getattr(conf, 'escnn_ekf_alpha', 0.99)
        sigma_p0 = getattr(conf, 'escnn_ekf_sigma_p0', 0.1)
        sigma_q = getattr(conf, 'escnn_ekf_sigma_q', 0.01)
        sigma_r = getattr(conf, 'escnn_ekf_sigma_r', 0.5)
        chunk_size = getattr(conf, 'escnn_ekf_jacobian_chunk_size', 16)
        if not hasattr(self, '_ekf_trackers'):
            self._ekf_trackers = [[EkfParamTracker(net, dynamics, alpha, sigma_p0, sigma_q, sigma_r, chunk_size)
                                    for net in nets] for nets in self.detector]
        else:
            for nets, tracker_row in zip(self.detector, self._ekf_trackers):
                for net, tracker in zip(nets, tracker_row):
                    tracker.rebind(net)
        return self._ekf_trackers

    def ekf_predict_only(self):
        """EKF time step alone, at the START of a group (ekf.py's ekfcrc/ekfht): predict every
        (user, iteration) tracker - theta <- alpha*theta + (1-alpha)*theta_pretrained, P <- alpha^2 P + Q -
        and write the predicted theta into the networks, so the group's detection runs with the prior
        theta_{t|t-1} (proper predict -> detect -> update order). The updates that follow are then
        called with do_predict=False. No-op when escnn_load_freeze leaves nothing trainable."""
        if not any(p.requires_grad for nets in self.detector for net in nets for p in net.parameters()):
            return
        for tracker_row in self._get_ekf_trackers():
            for tracker in tracker_row:
                tracker.predict()
                tracker._write_back()

    def ekf_predict_update(self, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int,
                            probs_in: torch.Tensor = None, payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT,
                            tx_ref: torch.Tensor = None, use_bp: bool = False, crc_check_fn=None,
                            bp_gate: str = 'none'):
        """Unsupervised, syndrome-driven test-time adaptation: for every (user, iteration)
        network, whatever escnn_load_freeze leaves unfrozen gets one EKF predict+update per
        *slot* (one TB/LDPC codeword each) from that slot's soft-syndrome measurement (no
        ground-truth bits), written back into the network in place. No pilot/val split - the
        caller decides what rx_real covers (a whole block, a single slot, a group of slots -
        see ekf.py). probs_in mirrors _forward's augmentation input: used when
        conf.which_augment != 'NO_AUGMENT', otherwise the flat 0.5 init is used.
        payload_symbols_per_slot is how many of rx_real's rows make up one slot - defaults to a
        full NUM_SYMB_PER_SLOT, but ekf.py's streaming-drift script embeds DMRS in-slot and
        passes its own payload-symbols-per-slot count instead, since rx_real there already
        excludes DMRS rows (see _get_syndrome_helper's docstring for why this must also match
        the codec that built the actual LDPC codewords). See ekf_syndrome.tex.

        tx_ref (diagnostic only): the same slots' true bits, (symbols*qm, n_users, num_res), llr > 0
        <=> bit 1. When given and logging is on, each update also computes the step those true bits
        would have produced (ekfi's agreement measurement, same prior) and logs cos_true, its
        cosine similarity with the applied syndrome step. Never applied; doubles the per-slot cost."""
        helper = self._get_syndrome_helper(tag='ekf', payload_symbols_per_slot=payload_symbols_per_slot)
        if helper is None:
            self._tsyn_warn_once('ekf_mcs', "EKF tracking needs conf.mcs > -1 (LDPC); skipping EKF update.", tag='ekf')
            return
        if not getattr(conf, 'encode_pilots', False) or conf.make_64QAM_16QAM_percentage != 0:
            self._tsyn_warn_once('ekf_encode_pilots', "EKF tracking needs encode_pilots: True and "
                                  "make_64QAM_16QAM_percentage: 0 (LDPC structure required); skipping EKF update.", tag='ekf')
            return

        # When conf.mod_pilot pads the network to a wider num_bits than the data's real qm (see
        # ekf.py's num_bits_pilot docstring), net(...)'s LLR output carries num_bits channels per
        # symbol, only qm of which are real (the rest are the LLR=0/prob=0.5 padding channels).
        # The syndrome measurement below must select the same real_bit_idx channels ekf.py's own
        # BER scoring does before handing the stream to helper.p_vector() - otherwise the
        # flattened prefix mixes in uninformative padding channels (or, when num_bits/qm divides
        # evenly, silently drops later symbols instead), making the measurement structurally
        # meaningless regardless of how good the real LLRs are.
        qm = getattr(self, '_synd_qm', num_bits)
        real_bit_idx = None
        if num_bits != qm:
            real_bit_idx = torch.as_tensor(relevant_indices(num_bits, num_bits / qm),
                                            device=DEVICE, dtype=torch.long)

        # Slot count from the actual symbol-domain slicing below
        # (rx_prob[s*payload_symbols_per_slot:...]), not from a bits-available/helper.n estimate:
        # rx_real here is pilot-domain (num_bits == num_bits_pilot), while helper.n is sized from
        # the data qm (see ekf.py's ldpc_n). When mod_pilot's bit depth differs from the data
        # MCS's (e.g. mod_pilot=16 -> num_bits_pilot=4 vs MCS2's qm=2), that estimate is off by
        # the qm ratio and can claim more slots than the tensor actually holds, indexing empty
        # slices in the per-slot loops below.
        num_slots = rx_real.shape[0] // payload_symbols_per_slot
        if num_slots == 0:
            self._tsyn_warn_once('ekf_too_small', f"block has only {rx_real.shape[0]} symbols - too "
                                  f"small for one codeword (needs {helper.n} bits); skipping EKF update.", tag='ekf')
            return
        # escnn_load_freeze='all' (or any mode that happens to leave nothing trainable) means
        # there's nothing for the EKF to track - bypass it entirely rather than letting
        # EkfParamTracker raise, so this becomes a legitimate "run the loaded weights
        # statically, no adaptation" mode instead of a crash.
        if not any(p.requires_grad for nets in self.detector for net in nets for p in net.parameters()):
            self._tsyn_warn_once('ekf_all_frozen', "escnn_load_freeze leaves nothing trainable - "
                                  "skipping EKF update (running loaded weights statically).", tag='ekf')
            if getattr(conf, 'log_train_every_epochs', 0) > 0:
                self._log_static_syndrome_stats(rx_real, num_bits, n_users, iterations, probs_in, helper,
                                                 num_slots, real_bit_idx, payload_symbols_per_slot)
            return

        trackers = self._get_ekf_trackers()
        rx_real = rx_real.to(DEVICE).unsqueeze(-1)  # (symbols, C, num_res) -> (symbols, C, num_res, 1), matching probs_vec / _forward's rx.unsqueeze(-1)
        # ekf.py weights_track_mode='ekfbp': measurement = ALL check rows, punctured columns filled
        # per row from a detached BP decoder run on the network's own LLRs (BP's extrinsic message
        # L_{v->m}, i.e. v's estimate from every row except m) instead of one peeling round + clean
        # rows. Transmitted columns keep the network's t(theta) - the only part with a Jacobian.
        # Validated in the ekfi ladder as step 'ekfibp' (tracks like the genie R3).
        bp_punc_edge = None
        # bp_gate (set by ekf.py's mode: 'ekfbp' none, 'ekfbps' 'syndrome', 'ekfbpc' 'crc'): skip a slot's
        # update (predict still runs) unless the slot looks decodable - 'syndrome': the hard decision of
        # the detached BP's posterior satisfies every parity check; 'crc': the Sionna LDPC decode of the
        # prior-weight LLRs passes CRC (crc_check_fn) - see _gate_check.
        bp_gate = self._bp_gate_mode(bp_gate, use_bp, crc_check_fn)
        if use_bp:
            helper._to(DEVICE)
            _is_p = torch.zeros(helper.n_ldpc, dtype=torch.bool, device=helper.edge_bit_all.device)
            _is_p[helper.punctured_idx] = True
            bp_punc_edge = _is_p[helper.edge_bit_all]
        if getattr(conf, 'which_augment', 'NO_AUGMENT') == 'NO_AUGMENT' or probs_in is None:
            probs_vec = self._initialize_probs_for_infer(rx_real, num_bits, n_users)
        else:
            probs_vec = probs_in.to(DEVICE)
        no_samples = getattr(conf, 'no_samples', False)
        log_stats = getattr(conf, 'log_train_every_epochs', 0) > 0
        use_ref = log_stats and tx_ref is not None

        for i in range(iterations):
            next_probs_vec = probs_vec.clone()
            for user in range(n_users):
                net = self.detector[user][i]
                if conf.no_probs:
                    rx_prob = rx_real
                elif no_samples:
                    rx_prob = probs_vec
                else:
                    rx_prob = torch.cat((rx_real, probs_vec), dim=1)
                if use_ref:
                    # +-1 per real bit, (symbols, qm, num_res)
                    sign_ref = (2.0 * tx_ref[:, user, :].reshape(-1, qm, rx_real.shape[2]) - 1.0).to(
                        DEVICE, dtype=rx_prob.dtype)

                # One slot = one TB = one LDPC codeword here (this project has no CB
                # segmentation), and slots are the actual unit of elapsed transmission
                # time - unlike "block", whose slot count is just an artifact of
                # pilot_size/block_length_factor bookkeeping. predict() runs once per
                # group of slots_per_group consecutive slots (one Kalman
                # "epoch" = however many slots are assumed to share one channel
                # realization - see conf.slots_per_group), injecting process
                # noise / the ar1 pull once per group; the slots within a group are then
                # each a sequential update() at that same epoch (no re-predicting between
                # them), matching the earlier reasoning: predict models elapsed time
                # between epochs, multiple simultaneous measurements at one epoch don't
                # each get their own. Default of 1 reproduces one predict per slot.
                # ESCNNDetector's Conv2d kernels only span (kernel_size, 1) over the
                # num_res axis - the symbol/batch axis is never convolved across - so a
                # slot's LLRs depend only on that slot's own payload_symbols_per_slot symbols.
                # Slicing rx_prob down to just those symbols before the forward pass (not
                # after) gives identical results while avoiding redoing other slots'
                # worth of conv work on every one of the updates below.
                tracker = trackers[user][i]
                slots_per_predict = max(1, int(getattr(conf, 'slots_per_group', 1)))
                for group_start in range(0, num_slots, slots_per_predict):
                    group_end = min(group_start + slots_per_predict, num_slots)
                    tracker.predict()
                    for s in range(group_start, group_end):
                        rx_slot = rx_prob[s * payload_symbols_per_slot:(s + 1) * payload_symbols_per_slot]

                        bp_tpe, bp_diag, bp_unsat, gate_txt = None, "", None, ""
                        if use_bp:
                            with torch.no_grad():
                                _, _llr_bp = functional_call(net, tracker._split(tracker.theta), (rx_slot,))
                                if real_bit_idx is not None:
                                    _llr_bp = _llr_bp[:, real_bit_idx, :, :]
                                _Lbp = _llr_bp.squeeze(-1).reshape(-1)[:helper.n]
                                _v2c, _post = self._ladder_bp(helper, _Lbp)
                                bp_tpe = torch.tanh(_v2c.clamp(-30.0, 30.0) / 2.0)
                                if bp_gate != 'none':
                                    bp_unsat, gate_txt = self._gate_check(bp_gate, helper, _post, _Lbp, crc_check_fn)
                                if use_ref:
                                    # Diagnostic only (true bits never enter the measurement).
                                    _sg = sign_ref[s * payload_symbols_per_slot:(s + 1) * payload_symbols_per_slot]
                                    _lt = 30.0 * _sg.reshape(1, -1)[:, :helper.n]
                                    _tt = torch.tanh(helper.map_to_mother(_lt).clamp(-30.0, 30.0) / 2.0)
                                    _flag = helper._fallback_rounds_logged
                                    helper._fallback_rounds_logged = True
                                    _tt = helper._estimate_punctured_t(_tt, 50)[0]
                                    helper._fallback_rounds_logged = _flag
                                    _pi, _ti = helper.punctured_idx, helper.tx_to_mother
                                    _known = _tt[_pi].abs() > 0.5
                                    _nk = max(int(_known.sum().item()), 1)
                                    _pw = (((_post[_pi] * _tt[_pi]) < 0) & _known).sum().item() / _nk
                                    _tw = ((_post[_ti] * _tt[_ti]) < 0).float().mean().item()
                                    _rw = ((_Lbp > 0) != (_sg.reshape(-1)[:helper.n] > 0)).float().mean().item()
                                    bp_diag = (f" tx_wrong={_rw:.4f} bp_punc_wrong={_pw:.4f}"
                                               f" bp_tx_wrong={_tw:.4f}")
                                bp_diag += gate_txt

                        def measurement_fn(param_dict, _net=net, _rx=rx_slot, _n=helper.n, _idx=real_bit_idx,
                                           _tpe=bp_tpe):
                            _, llrs = functional_call(_net, param_dict, (_rx,))
                            if _idx is not None:
                                llrs = llrs[:, _idx, :, :]
                            stream = llrs.squeeze(-1).reshape(-1)[:_n].reshape(1, _n)
                            if _tpe is not None:
                                t_col = torch.tanh(helper.map_to_mother(stream).clamp(-30.0, 30.0) / 2.0)
                                te = torch.where(bp_punc_edge, _tpe, t_col[0, helper.edge_bit_all])
                                return self._ladder_row_products(te, helper.edge_check_all, helper.num_checks)
                            return helper.p_vector(stream).reshape(-1)

                        reference_fn = None
                        if use_ref:
                            _sign = sign_ref[s * payload_symbols_per_slot:(s + 1) * payload_symbols_per_slot]

                            def reference_fn(param_dict, _net=net, _rx=rx_slot, _idx=real_bit_idx, _sign=_sign):
                                _, llrs = functional_call(_net, param_dict, (_rx,))
                                if _idx is not None:
                                    llrs = llrs[:, _idx, :, :]
                                return (_sign * torch.tanh(0.5 * llrs.squeeze(-1))).reshape(-1)

                        if bp_unsat is not None and bp_unsat > 0:
                            # Gated: BP did not reach a valid codeword, so its punctured values are not
                            # trusted - keep the predicted state (push it into the net) and skip the update.
                            tracker._write_back()
                            if log_stats:
                                print(f"[gate] user={user} it={i} slot={s + 1}/{num_slots} "
                                      f"update skipped ({bp_gate} gate){bp_diag or gate_txt}", flush=True)
                            continue
                        stats = tracker.update(measurement_fn, reference_fn=reference_fn)
                        if log_stats and not stats.get('skipped', True):
                            cos_txt = (f" cos_true={stats['cos_ref']:.3f}"
                                       if stats.get('cos_ref') is not None else "")
                            print(f"[ekf] user={user} it={i} slot={s + 1}/{num_slots} "
                                  f"(epoch {group_start // slots_per_predict + 1}) "
                                  f"checks={stats['num_checks']} mean_hard_sat={stats['mean_hard_sat']:.3f} "
                                  f"mean_p={stats['mean_p']:.3f} dtheta_rms={stats['dtheta_rms']:.3e}{cos_txt}{bp_diag}",
                                  flush=True)

                with torch.no_grad():
                    output, _ = net(rx_prob)
                    index_start = user * num_bits
                    index_end = (user + 1) * num_bits
                    next_probs_vec[:, index_start:index_end, :, :] = output
            probs_vec = next_probs_vec

    def ekf_predict_update_supervised(self, rx_real: torch.Tensor, tx: torch.Tensor, num_bits: int, n_users: int,
                                       iterations: int, probs_in: torch.Tensor = None,
                                       payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT,
                                       ekfi_step: int = 0, update_mask=None, log_suffix: str = "",
                                       crc_check_fn=None, bp_gate: str = 'none', fail_step: int = None,
                                       fail_log_suffix: str = "", tx_diag: torch.Tensor = None,
                                       do_predict: bool = True):
        """Supervised counterpart of ekf_predict_update (ekf.py's weights_track_mode='ekfi'): the
        same EkfParamTracker predict/update, but the measurement is the known transmitted bits of a
        separate calibration region instead of the soft syndrome - the single-step CM-EKF of
        Gusakov et al. (IEEE TSP 2026) applied to ESCNN. One predict per call (= per group), then
        one update per calibration slot. Impractical in the same sense as sgdbcei: it needs a whole
        known, extra region every group.

        Per-bit measurement is the agreement (2b-1)*tanh(L/2) in [-1, 1] with target 1, i.e. the
        same "1 - p" innovation form the tracker already uses for the syndrome. Since
        tanh(L/2) = 2*sigmoid(L) - 1, the innovation is 2*|b - sigmoid(L)|: the paper's
        b - h(x), scaled by 2.

        tx: (symbols*num_bits, n_users, num_res) bits, the same layout sgdbcei passes to
        _online_training (llr > 0 <=> bit 1, as in BCEWithLogitsLoss).

        DEBUG-LADDER (remove after the ekfi -> ekf debug) - ekfi_step selects the measurement,
        one change per step (ekf.py's 'ekfi' = R0, 'ekfi1'..'ekfi5' = R1..R5; R6 never gets here):
          R0: agreement over all bits of the slot, no clamp (the original ekfi measurement)
          R1: + LLRs clamped to +-LLR_CLAMP (30), as in the syndrome measurement
          R2: + only the bits covered by >= 1 clean check (helper.edge_bit_clean)
          R3: check rows p_m instead of bit rows, punctured bits filled from the TRUE codeword
              (erasure peeling seeded with +-30 LLRs of the known bits - same rounds as ekf's
              peeling), all non-dead checks. Data term becomes sum_m (1-p_m)^2.
          R4: + only the clean checks
          R5: punctured bits from peeling the network's own LLRs = helper.p_vector,
              i.e. exactly ekf's measurement, still on the calib slots.
          41 (R4s): R4's clean rows, punctured columns = TRUE sign x network-peeled magnitude
          42 (R4m): R4's clean rows, punctured columns = network-peeled sign x TRUE magnitude (~1)
          50 (RBP): ALL check rows. A detached sum-product BP decoder (_LADDER_BP_ITERS iterations)
              runs over every column from the network's own LLRs; each punctured column v in row m
              then gets BP's extrinsic message L_{v->m} (v's estimate from every row except m), so no
              row is scored against a value it produced itself and no row is dead. Transmitted
              columns keep the network's t(theta) - the only part that carries the Jacobian.

        update_mask (optional, (num_slots, n_users) bool): slots/users whose update is skipped where
        False (the group's predict still runs) - ekf.py's 'ekfcrc' passes its CRC-pass mask here, with
        tx holding the re-encoded decoded codewords. bp_gate (default 'none') likewise skips step 50's update -
        'syndrome': the BP posterior's hard decision fails a parity check; 'crc': the Sionna decode of
        the prior-weight LLRs fails CRC (crc_check_fn). log_suffix is appended to
        each [ekfi] line.

        fail_step (optional, ekf.py's 'ekfht'): instead of skipping the slots/users update_mask marks
        False, update them with ladder step fail_step's measurement (ekfht: 50 = the BP syndrome
        measurement, which never reads tx) - so one predict, then one update per slot from whichever
        source that slot has. fail_log_suffix replaces log_suffix on those slots. tx_diag (optional,
        same layout as tx): the true bits, used only for the log diagnostics (cos_true, tx_wrong,
        bp_*_wrong) in place of tx - needed when tx holds dummy labels on the fail_step slots.
        do_predict=False skips this call's predict() - for a second update pass on the same group/epoch
        (ekf.py's ekfht label update after re-decoding), so no extra Q / ar1 pull is applied.

        Diagnostics when log_train_every_epochs > 0: cos_true per update - cosine between the applied
        step and the step R0's measurement (true-bit agreement over every bit of the same calib
        slot) would take from the same prior, i.e. the same reference ekf's own cos_true uses; and,
        once per run for steps >= 2, how many clean / non-dead rows each transmitted column is in."""
        if not any(p.requires_grad for nets in self.detector for net in nets for p in net.parameters()):
            self._tsyn_warn_once('ekfi_all_frozen', "escnn_load_freeze leaves nothing trainable - "
                                  "skipping EKF update (running loaded weights statically).", tag='ekf')
            return
        num_slots = rx_real.shape[0] // payload_symbols_per_slot
        if num_slots == 0:
            return

        # DEBUG-LADDER: steps >= 2 need the same SyndromeLoss helper ekf uses (same code, same
        # clean/dead classification, same peeling rounds).
        step = int(ekfi_step)
        fail_step = None if fail_step is None else int(fail_step)
        if fail_step is not None and fail_step not in (0, 1, 5, 50):
            raise ValueError(f"fail_step={fail_step} - only steps whose measurement never reads tx "
                             f"(50 = BP syndrome, 5 = peeled syndrome) or plain label steps (0, 1) are allowed.")
        setup_steps = {step} if fail_step is None else {step, fail_step}   # steps whose setup must exist
        helper, tx_keep, punc_in_clean, bp_punc_edge = None, None, None, None
        if max(setup_steps) >= 2:
            helper = self._get_syndrome_helper(tag='ekf', payload_symbols_per_slot=payload_symbols_per_slot)
            if helper is None:
                raise ValueError(f"ekfi step R{step} needs the LDPC syndrome helper (conf.mcs > -1).")
            helper._to(DEVICE)
            if step == 2:
                # Plain mask instead of torch.isin (isin crashed with SIGILL on cluster node ise-cpu-intl-15).
                _in_clean = torch.zeros(helper.n_ldpc, dtype=torch.bool, device=helper.edge_bit_clean.device)
                _in_clean[helper.edge_bit_clean] = True
                tx_keep = _in_clean[helper.tx_to_mother]  # (n,) bool
            # Diagnostic: which punctured columns appear in >= 1 clean row (the ones R4/R5/ekf use).
            _in_clean_p = torch.zeros(helper.n_ldpc, dtype=torch.bool, device=helper.edge_bit_clean.device)
            _in_clean_p[helper.edge_bit_clean] = True
            punc_in_clean = _in_clean_p[helper.punctured_idx]   # (P,) bool
            if 50 in setup_steps:
                _is_p = torch.zeros(helper.n_ldpc, dtype=torch.bool, device=helper.edge_bit_all.device)
                _is_p[helper.punctured_idx] = True
                bp_punc_edge = _is_p[helper.edge_bit_all]          # (E,) bool
            if getattr(conf, 'log_train_every_epochs', 0) > 0 and not getattr(self, '_ladder_cov_logged', False):
                self._log_ladder_row_coverage(helper)
                self._ladder_cov_logged = True
        # Same +-30 as SyndromeLoss.LLR_CLAMP (not imported here: syndrome_loss imports Sionna lazily).
        llr_clamp = (helper.LLR_CLAMP if helper is not None else 30.0) if max(setup_steps) >= 1 else None
        bp_gate = self._bp_gate_mode(bp_gate, step == 50, crc_check_fn)   # gate: primary step only

        trackers = self._get_ekf_trackers()
        rx_real = rx_real.to(DEVICE).unsqueeze(-1)
        if getattr(conf, 'which_augment', 'NO_AUGMENT') == 'NO_AUGMENT' or probs_in is None:
            probs_vec = self._initialize_probs_for_infer(rx_real, num_bits, n_users)
        else:
            probs_vec = probs_in.to(DEVICE)
        no_samples = getattr(conf, 'no_samples', False)
        log_stats = getattr(conf, 'log_train_every_epochs', 0) > 0
        num_res = rx_real.shape[2]

        for i in range(iterations):
            next_probs_vec = probs_vec.clone()
            for user in range(n_users):
                net = self.detector[user][i]
                if conf.no_probs:
                    rx_prob = rx_real
                elif no_samples:
                    rx_prob = probs_vec
                else:
                    rx_prob = torch.cat((rx_real, probs_vec), dim=1)
                # +-1 per bit, (symbols, num_bits, num_res)
                sign = (2.0 * tx[:, user, :].reshape(-1, num_bits, num_res) - 1.0).to(DEVICE, dtype=rx_prob.dtype)
                # Diagnostics-only sign (true bits when tx_diag is given, else the labels themselves).
                sign_d = sign if tx_diag is None else (
                    2.0 * tx_diag[:, user, :].reshape(-1, num_bits, num_res) - 1.0).to(DEVICE, dtype=rx_prob.dtype)

                tracker = trackers[user][i]
                if do_predict:
                    tracker.predict()
                for s in range(num_slots):
                    sl = slice(s * payload_symbols_per_slot, (s + 1) * payload_symbols_per_slot)
                    rx_slot, sign_slot, sign_dslot = rx_prob[sl], sign[sl], sign_d[sl]
                    masked = update_mask is not None and not bool(update_mask[s][user])
                    # ekfht: masked slots (no CRC pass -> no labels) fall back to fail_step's measurement.
                    slot_step = fail_step if (masked and fail_step is not None) else step
                    sfx = fail_log_suffix if (masked and fail_step is not None) else log_suffix
                    slot_label = {41: '4s', 42: '4m', 50: 'BP'}.get(slot_step, str(slot_step))
                    if masked and fail_step is None:
                        # e.g. ekfcrc: this slot's decode failed CRC - no labels, keep the predicted state.
                        tracker._write_back()
                        if log_stats:
                            print(f"[gate] user={user} it={i} calib slot={s + 1}/{num_slots} "
                                  f"update skipped (masked){log_suffix}", flush=True)
                        continue

                    # DEBUG-LADDER R3/R4: punctured values from the TRUE codeword (constant in theta)
                    # DEBUG-LADDER RBP: detached BP on the network's own LLRs (prior weights), giving
                    # per-edge punctured values tanh(L_{v->m}/2), constant w.r.t. theta.
                    bp_tpe, bp_diag, bp_unsat, gate_txt = None, "", None, ""
                    if slot_step == 50:
                        with torch.no_grad():
                            _, _llr_bp = functional_call(net, tracker._split(tracker.theta), (rx_slot,))
                            _Lbp = _llr_bp.squeeze(-1).reshape(-1)[:helper.n]
                            _v2c, _post = self._ladder_bp(helper, _Lbp)
                            bp_tpe = torch.tanh(_v2c.clamp(-30.0, 30.0) / 2.0)
                            if bp_gate != 'none' and slot_step == step:
                                bp_unsat, gate_txt = self._gate_check(bp_gate, helper, _post, _Lbp, crc_check_fn)
                            if log_stats:
                                # Truth on every mother-code column (peeling the true bits to a fixpoint).
                                _lt = 30.0 * sign_dslot.reshape(1, -1)[:, :helper.n]
                                _tt = torch.tanh(helper.map_to_mother(_lt).clamp(-30.0, 30.0) / 2.0)
                                _flag = helper._fallback_rounds_logged
                                helper._fallback_rounds_logged = True       # keep ekf's own log line intact
                                _tt = helper._estimate_punctured_t(_tt, 50)[0]
                                helper._fallback_rounds_logged = _flag
                                _pi = helper.punctured_idx
                                _known = _tt[_pi].abs() > 0.5
                                _nk = max(int(_known.sum().item()), 1)
                                _pw = (((_post[_pi] * _tt[_pi]) < 0) & _known).sum().item() / _nk
                                _ti = helper.tx_to_mother
                                _tw = ((_post[_ti] * _tt[_ti]) < 0).float().mean().item()
                                bp_diag = (f" bp_punc_wrong={_pw:.4f} bp_tx_wrong={_tw:.4f}"
                                           f" bp_punc_known={_nk}")

                    t_punc_true, nondead = None, None
                    if slot_step in (3, 4, 41, 42):
                        with torch.no_grad():
                            llr_true = llr_clamp * sign_slot.reshape(1, -1)[:, :helper.n]
                            t_true = torch.tanh(helper.map_to_mother(llr_true).clamp(-llr_clamp, llr_clamp) / 2.0)
                            t_true = helper._estimate_punctured_t(t_true, helper.fallback_iters)
                            t_punc_true = t_true[:, helper.punctured_idx]
                            p_true = helper._check_products(t_true, helper.edge_check_all,
                                                            helper.edge_bit_all, helper.num_checks)
                            nondead = p_true[0].abs() > 0.5   # resolved checks: p = +1; dead: 0

                    def measurement_fn(param_dict, _net=net, _rx=rx_slot, _sign=sign_slot,
                                       _tp=t_punc_true, _nd=nondead, _tpe=bp_tpe, _st=slot_step):
                        step = _st   # this slot's step (ekfht: labels on CRC-pass slots, fail_step otherwise)
                        _, llrs = functional_call(_net, param_dict, (_rx,))
                        L = llrs.squeeze(-1).reshape(-1)               # L > 0 <=> bit 1
                        sgn = _sign.reshape(-1)
                        if step == 0:                                    # R0
                            return sgn * torch.tanh(0.5 * L)
                        L = L.clamp(-llr_clamp, llr_clamp)
                        if step == 1:                                    # R1
                            return sgn * torch.tanh(0.5 * L)
                        if step == 2:                                    # R2
                            return (sgn[:helper.n] * torch.tanh(0.5 * L[:helper.n]))[tx_keep]
                        stream = L[:helper.n].reshape(1, -1)
                        if step == 50:                                   # RBP: all rows, BP punctured
                            t_col = torch.tanh(helper.map_to_mother(stream).clamp(-llr_clamp, llr_clamp) / 2.0)
                            te = t_col[0, helper.edge_bit_all]
                            te = torch.where(bp_punc_edge, _tpe, te)
                            return self._ladder_row_products(te, helper.edge_check_all, helper.num_checks)
                        if step == 5:                                    # R5 (= ekf's measurement)
                            return helper.p_vector(stream).reshape(-1)
                        t = torch.tanh(helper.map_to_mother(stream).clamp(-llr_clamp, llr_clamp) / 2.0).clone()
                        if step in (41, 42):
                            # Network-peeled punctured values, detached exactly as in p_vector (R5).
                            t_net_p = helper._estimate_punctured_t(t.detach(), helper.fallback_iters)[
                                :, helper.punctured_idx]
                            if step == 41:                               # R4s: true sign, network magnitude
                                t[:, helper.punctured_idx] = torch.sign(_tp) * t_net_p.abs()
                            else:                                        # R4m: network sign, true magnitude
                                t[:, helper.punctured_idx] = torch.sign(t_net_p) * _tp.abs()
                            return helper._check_products(t, helper.edge_check_clean, helper.edge_bit_clean,
                                                          helper.num_clean).reshape(-1)
                        t[:, helper.punctured_idx] = _tp
                        if step == 4:                                    # R4
                            return helper._check_products(t, helper.edge_check_clean, helper.edge_bit_clean,
                                                          helper.num_clean).reshape(-1)
                        p = helper._check_products(t, helper.edge_check_all, helper.edge_bit_all,
                                                   helper.num_checks)    # R3
                        return p[:, _nd].reshape(-1)

                    # DEBUG-LADDER diagnostic: R0's true-bit agreement on this same calib slot (all
                    # bits, no clamp) - the same reference ekf's cos_true uses. Never applied.
                    reference_fn = None
                    if log_stats:
                        def reference_fn(param_dict, _net=net, _rx=rx_slot, _sign=sign_dslot):
                            _, llrs = functional_call(_net, param_dict, (_rx,))
                            return (_sign.reshape(-1) * torch.tanh(0.5 * llrs.squeeze(-1).reshape(-1)))

                    # DEBUG-LADDER diagnostic, prior weights: sign errors of the network's own LLRs on
                    # this calib slot (tx_wrong), and of the punctured values peeled from them, over the
                    # punctured columns used by clean rows (punc_wrong, vs the true codeword; punc_unres =
                    # fraction left unresolved). Read-only.
                    diag_txt = ""
                    if log_stats:
                        with torch.no_grad():
                            # Same prior weights the update linearizes at (tracker.theta after predict).
                            _, _llr0 = functional_call(net, tracker._split(tracker.theta), (rx_slot,))
                            _L0 = _llr0.squeeze(-1).reshape(-1)
                            _sg = sign_dslot.reshape(-1)
                            diag_txt = f" tx_wrong={((_L0 > 0) != (_sg > 0)).float().mean().item():.4f}"
                            if helper is not None:
                                _L0n = _L0[:helper.n].clamp(-30.0, 30.0).reshape(1, -1)
                                _t0 = torch.tanh(helper.map_to_mother(_L0n).clamp(-30.0, 30.0) / 2.0)
                                _pn = helper._estimate_punctured_t(_t0, helper.fallback_iters)[0, helper.punctured_idx]
                                _lt = 30.0 * _sg.reshape(1, -1)[:, :helper.n]
                                _tt = torch.tanh(helper.map_to_mother(_lt).clamp(-30.0, 30.0) / 2.0)
                                _pt = helper._estimate_punctured_t(_tt, helper.fallback_iters)[0, helper.punctured_idx]
                                _m = punc_in_clean & (_pt.abs() > 0.5)
                                _nm = max(int(_m.sum().item()), 1)
                                _wrong = ((_pn * _pt) < 0) & _m
                                _unres = (_pn.abs() < 1e-12) & _m
                                diag_txt += (f" punc_wrong={_wrong.sum().item() / _nm:.4f}"
                                             f" punc_unres={_unres.sum().item() / _nm:.4f} punc_n={_nm}")

                    diag_txt += bp_diag
                    if bp_unsat is not None:
                        diag_txt += gate_txt
                        if bp_unsat > 0:
                            tracker._write_back()
                            if log_stats:
                                print(f"[gate] user={user} it={i} calib slot={s + 1}/{num_slots} "
                                      f"update skipped ({bp_gate} gate){diag_txt}{sfx}", flush=True)
                            continue
                    stats = tracker.update(measurement_fn, reference_fn=reference_fn)
                    if log_stats and not stats.get('skipped', True):
                        # Prefix stays "[ekfi]" (plot_drift_log.py's DTHETA_RE); step appended for R1..R5.
                        cos_txt = (f" cos_true={stats['cos_ref']:.3f}"
                                   if stats.get('cos_ref') is not None else "")
                        print(f"[ekfi] user={user} it={i} calib slot={s + 1}/{num_slots} "
                              f"bits={stats['num_checks']} frac_correct={stats['mean_hard_sat']:.3f} "
                              f"mean_agree={stats['mean_p']:.3f} dtheta_rms={stats['dtheta_rms']:.3e}"
                              f"{cos_txt}{diag_txt}" + (f" step=R{slot_label}" if slot_step else "") + sfx,
                              flush=True)

                with torch.no_grad():
                    output, _ = net(rx_prob)
                    next_probs_vec[:, user * num_bits:(user + 1) * num_bits, :, :] = output
            probs_vec = next_probs_vec

    _LADDER_BP_ITERS = 10   # DEBUG-LADDER RBP: default sum-product iterations (config: bp_iters)

    @torch.no_grad()
    def _ladder_bp(self, helper, llr_tx: torch.Tensor, iters: int = None):
        """DEBUG-LADDER RBP: detached flooding sum-product BP over the full mother code.
        llr_tx: (n,) project-convention LLRs (L>0 <=> bit 1) of one codeword. Channel LLRs: transmitted
        columns from llr_tx, fillers +30, punctured 0 (erasures). Returns (v2c, post): the per-edge
        variable-to-check messages after the last iteration (classical convention, L>0 <=> bit 0) -
        v2c[e] for edge (m, v) is v's estimate from every row except m - and the posterior LLR per
        mother-code column."""
        if iters is None:
            iters = getattr(conf, 'bp_iters', None)
        iters = self._LADDER_BP_ITERS if iters is None else int(iters)
        ec, eb = helper.edge_check_all, helper.edge_bit_all
        Lch = helper.map_to_mother(llr_tx.clamp(-30.0, 30.0).reshape(1, -1))[0]
        v2c = Lch[eb].clone()
        post = Lch.clone()
        M = helper.num_checks
        for _ in range(iters):
            t = torch.tanh(v2c.clamp(-30.0, 30.0) / 2.0)
            la = torch.log(t.abs().clamp(min=1e-12, max=1.0 - 1e-7))
            ng = (t < 0).to(t.dtype)
            sla = torch.zeros(M, dtype=t.dtype, device=t.device).index_add_(0, ec, la)
            sng = torch.zeros(M, dtype=t.dtype, device=t.device).index_add_(0, ec, ng)
            prod = (1.0 - 2.0 * torch.remainder(sng[ec] - ng, 2.0)) * torch.exp(sla[ec] - la)
            c2v = 2.0 * torch.atanh(prod.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            post = Lch.clone().index_add_(0, eb, c2v)
            v2c = post[eb] - c2v
        return v2c, post

    _BP_GATES = ('none', 'syndrome', 'crc')

    def _bp_gate_mode(self, gate: str, bp_active: bool, crc_check_fn=None) -> str:
        """The requested gate, validated and reduced to 'none' where it cannot apply here."""
        gate = str(gate or 'none').lower()
        if gate not in self._BP_GATES:
            raise ValueError(f"bp_gate={gate!r} not in {self._BP_GATES}.")
        if gate != 'none' and not bp_active:
            self._tsyn_warn_once('bp_gate_no_bp', f"bp_gate={gate!r} only applies to the BP modes "
                                  "- ignored here.", tag='ekf')
            return 'none'
        if gate == 'crc' and crc_check_fn is None:
            self._tsyn_warn_once('bp_gate_no_crc', "the CRC gate needs a CRC check callback from the "
                                  "caller - none given, gate disabled.", tag='ekf')
            return 'none'
        return gate

    def _gate_check(self, gate: str, helper, post: torch.Tensor, llr_tx: torch.Tensor, crc_check_fn):
        """Returns (n_fail, log_text); n_fail > 0 means skip this slot's update.
        'syndrome': unsatisfied checks of the BP posterior's hard decision (_bp_unsat).
        'crc': 1 if crc_check_fn(llr_tx as numpy, project convention L>0 <=> bit 1) reports a CRC
        failure of the Sionna LDPC decode of the prior-weight LLRs, else 0."""
        if gate == 'syndrome':
            n = self._bp_unsat(helper, post)
            return n, f" bp_unsat={n}"
        ok = bool(crc_check_fn(llr_tx.detach().cpu().numpy()))
        return (0 if ok else 1), f" crc_ok={int(ok)}"

    @staticmethod
    @torch.no_grad()
    def _bp_unsat(helper, post: torch.Tensor) -> int:
        """Number of unsatisfied parity checks (over all mother-code rows) of the hard decision of
        _ladder_bp's posterior (classical convention: post > 0 <=> bit 0). 0 = BP reached a valid
        codeword, so its punctured values can be trusted (mode 'ekfbps')."""
        x = (post < 0).to(post.dtype)
        par = torch.zeros(helper.num_checks, device=post.device, dtype=post.dtype).index_add_(
            0, helper.edge_check_all, x[helper.edge_bit_all])
        return int((torch.remainder(par, 2.0) > 0.5).sum().item())

    @staticmethod
    def _ladder_row_products(te: torch.Tensor, edge_check: torch.Tensor, num_checks: int) -> torch.Tensor:
        """Per-row product of per-edge t values (log domain, exact gradients) -> (num_checks,)."""
        log_abs = torch.log(te.abs().clamp(min=1e-30))
        sum_log = torch.zeros(num_checks, dtype=te.dtype, device=te.device).index_add(0, edge_check, log_abs)
        neg = torch.zeros(num_checks, dtype=te.dtype, device=te.device).index_add(0, edge_check, (te < 0).to(te.dtype))
        return (1.0 - 2.0 * torch.remainder(neg, 2.0)) * torch.exp(sum_log)

    @torch.no_grad()
    def _log_ladder_row_coverage(self, helper):
        """DEBUG-LADDER diagnostic, printed once: for every transmitted column, how many clean rows
        (R4/R5/ekf) and how many non-dead rows (R3) contain it. Structural only - non-dead rows are
        found by peeling an all-known input with the same round budget as the real peeling."""
        dev = helper.edge_bit_all.device
        t = torch.ones(1, helper.n_ldpc, device=dev)
        t[:, helper.punctured_idx] = 0.0
        t = helper._estimate_punctured_t(t, helper.fallback_iters)
        p = helper._check_products(t, helper.edge_check_all, helper.edge_bit_all, helper.num_checks)[0]
        nondead_rows = p.abs() > 0.5
        nd_edges = nondead_rows[helper.edge_check_all]
        cnt_nd = torch.bincount(helper.edge_bit_all[nd_edges], minlength=helper.n_ldpc)[helper.tx_to_mother]
        cnt_cl = torch.bincount(helper.edge_bit_clean, minlength=helper.n_ldpc)[helper.tx_to_mother]

        def summary(c):
            n = c.numel()
            return (f"0 rows {100.0 * (c == 0).sum().item() / n:.1f}%, 1 row {100.0 * (c == 1).sum().item() / n:.1f}%, "
                    f"2 rows {100.0 * (c == 2).sum().item() / n:.1f}%, >=3 rows {100.0 * (c >= 3).sum().item() / n:.1f}%, "
                    f"mean {c.float().mean().item():.2f}")
        print(f"[ekfi] row coverage of the {helper.n} transmitted columns - "
              f"clean rows ({helper.num_clean}): {summary(cnt_cl)} | "
              f"non-dead rows ({int(nondead_rows.sum().item())}): {summary(cnt_nd)}", flush=True)

    def _log_static_syndrome_stats(self, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int,
                                    probs_in: torch.Tensor, helper, num_slots: int, real_bit_idx=None,
                                    payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT):
        """Diagnostic-only counterpart to the tracked path's per-slot mean_hard_sat prints
        (ekf_predict_update() above), for when escnn_load_freeze leaves nothing trainable and that
        function returns before ever computing a measurement. Needed to compare "tracking on" vs
        "tracking off/static" mean_hard_sat under the same CFO drift - the frozen/static case is
        the natural baseline for checking the EKF is actually doing something, not the tracked
        case alone (see the plot_drift_log.py Kalman-convergence discussion).

        Mirrors ekf_predict_update()'s rx_prob construction and per-slot forward pass exactly, but
        read-only: no Jacobian, no predict/update, nothing written back - just measures how well
        the loaded (frozen) weights' own LLRs already satisfy the syndrome, slot by slot."""
        with torch.no_grad():
            rx_real = rx_real.to(DEVICE).unsqueeze(-1)
            if getattr(conf, 'which_augment', 'NO_AUGMENT') == 'NO_AUGMENT' or probs_in is None:
                probs_vec = self._initialize_probs_for_infer(rx_real, num_bits, n_users)
            else:
                probs_vec = probs_in.to(DEVICE)
            no_samples = getattr(conf, 'no_samples', False)

            for i in range(iterations):
                next_probs_vec = probs_vec.clone()
                for user in range(n_users):
                    net = self.detector[user][i]
                    if conf.no_probs:
                        rx_prob = rx_real
                    elif no_samples:
                        rx_prob = probs_vec
                    else:
                        rx_prob = torch.cat((rx_real, probs_vec), dim=1)

                    for s in range(num_slots):
                        rx_slot = rx_prob[s * payload_symbols_per_slot:(s + 1) * payload_symbols_per_slot]
                        _, llrs = net(rx_slot)
                        if real_bit_idx is not None:
                            llrs = llrs[:, real_bit_idx, :, :]
                        stream = llrs.squeeze(-1).reshape(-1)[:helper.n].reshape(1, helper.n)
                        p = helper.p_vector(stream).reshape(-1)
                        if p.numel() == 0:
                            continue
                        mean_hard_sat = float((p > 0).float().mean())
                        mean_p = float(p.mean())
                        print(f"[ekf] user={user} it={i} slot={s + 1}/{num_slots} (static) "
                              f"checks={p.numel()} mean_hard_sat={mean_hard_sat:.3f} "
                              f"mean_p={mean_p:.3f}", flush=True)

                    output, _ = net(rx_prob)
                    index_start = user * num_bits
                    index_end = (user + 1) * num_bits
                    next_probs_vec[:, index_start:index_end, :, :] = output
                probs_vec = next_probs_vec

    @staticmethod
    def _calc_h_marginal(est: torch.Tensor, mask: torch.Tensor = None) -> float:
        """H_b(q_bar) where q_bar = mean(sigmoid(L)) over non-pilot bits. Diagnostic only."""
        est_rs = est.squeeze(-1)
        q = torch.sigmoid(est_rs)
        if mask is not None:
            q = q[mask.bool()]
        q_bar = q.mean()
        eps = 1e-7
        h = -(q_bar * torch.log2(q_bar.clamp(min=eps))
              + (1.0 - q_bar) * torch.log2((1.0 - q_bar).clamp(min=eps)))
        return h.item()

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor, mask: torch.Tensor = None,
                         payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT) -> torch.Tensor:
        """
        Loss dispatched by conf.training_loss:
          'bce'  - BCEWithLogitsLoss (sigmoid applied internally); uses tx and mask.
          'gfmi' - Blind EXIT-MI loss: minimises 1-GFMI = mean binary entropy of |L|.
                   Accepts but ignores tx (kept for signature compatibility).
                   Mask is still applied so constant pilot bits are excluded.
          'tsyn' - tw * L_tent + (1-tw) * L_synd, where L_synd is a soft LDPC
                   syndrome penalty over the batch's codeword LLRs.

        payload_symbols_per_slot is forwarded to _syndrome_component (tsyn only) - see
        _train_model's docstring for why this must match whatever caller passed it in.
        """
        est_rs = est.squeeze(-1)
        loss_mode = getattr(conf, 'training_loss', 'bce')

        if loss_mode == 'gfmi':
            a = est_rs.abs()
            # log2(1 + exp(-|L|))  — numerically safe; → 0 for large |L|
            term1 = torch.log1p(torch.exp(-a)) / math.log(2)
            # (|L| / ln2) * sigmoid(-|L|)  — → 0 for large |L|
            term2 = (a / math.log(2)) * torch.sigmoid(-a)
            elementwise_loss = term1 + term2  # = H_b(sigma(|L|)), the per-bit GFMI term
        elif loss_mode in ('tent', 'tsyn'):
            # Prediction-entropy minimization (TENT, Wang et al. 2021).
            # H_b(sigma(L)) in bits via Bernoulli(logits=L).entropy(), which uses
            # softplus internally — stable at large |L|, no explicit sigmoid-then-log.
            # Algebraic equivalence: minimising tent == maximising GFMI
            # (same optimum, opposite sign convention; GFMI wraps as 1 - mean(H_b)).
            # The beta balance term below has no GFMI counterpart.
            # tsyn reuses this exact tent term unchanged as its L_tent component.
            elementwise_loss = torch.distributions.Bernoulli(logits=est_rs).entropy() / math.log(2)
        else:
            elementwise_loss = self.criterion(input=est_rs, target=tx)

        if mask is None:
            l0 = elementwise_loss.mean()
            q_flat = torch.sigmoid(est_rs)
        else:
            mask = mask.to(elementwise_loss.dtype)
            l0 = (elementwise_loss * mask).sum() / mask.sum().clamp(min=1.0)
            q_flat = torch.sigmoid(est_rs[mask.bool()])

        if loss_mode == 'tsyn':
            # NOTE: the two terms deliberately average over different bit sets —
            # L_tent over the full rate-matched detector grid, L_synd over the
            # code blocks (mother-codeword checks). This is intentional.
            l_tent = l0
            tw = float(getattr(conf, 'tw', 0.5))
            synd_out = self._syndrome_component(est_rs, payload_symbols_per_slot=payload_symbols_per_slot)
            if synd_out is None:
                self._tsyn_stats = {'l_tent': float(l_tent.detach()), 'l_synd': None, 'sat': None}
            else:
                l_synd, sat = synd_out
                l0 = tw * l_tent + (1.0 - tw) * l_synd
                if tw >= 1.0 and not getattr(self, '_tsyn_equiv_checked', False):
                    # tw=1 must reproduce training_loss='tent' exactly (first batch only)
                    diff = abs(float((l0 - l_tent).detach()))
                    assert diff < 1e-6, f"tsyn(tw=1) != tent: |diff|={diff:.3e}"
                    self._tsyn_equiv_checked = True
                self._tsyn_stats = {'l_tent': float(l_tent.detach()),
                                    'l_synd': float(l_synd.detach()), 'sat': sat}

        # Marginal-entropy balance term: loss -= beta * H_b(q_bar)
        # Maximises marginal entropy to push q_bar toward 0.5, preventing
        # all-same-sign collapse. Orthogonal to the tsyn mixing above.
        if loss_mode in ('gfmi', 'tent', 'tsyn'):
            beta = getattr(conf, 'beta_balance', 0.0)
        else:
            beta = 0.0
        if beta > 0.0:
            q_bar = q_flat.mean()
            eps = 1e-7
            h_marginal = -(q_bar * torch.log2(q_bar.clamp(min=eps))
                           + (1.0 - q_bar) * torch.log2((1.0 - q_bar).clamp(min=eps)))
            return l0 - beta * h_marginal
        return l0

    def _tsyn_warn_once(self, key: str, msg: str, tag: str = 'tsyn'):
        if not hasattr(self, '_tsyn_warned'):
            self._tsyn_warned = set()
        if key not in self._tsyn_warned:
            print(f"[{tag}] WARNING: {msg}", flush=True)
            self._tsyn_warned.add(key)

    def _get_syndrome_helper(self, tag: str = 'tsyn', payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT):
        """Lazily build/cache the SyndromeLoss for the current LDPC configuration
        (same k/n construction as evaluate.py's LDPC5GCodec, whose pilot region still spans a
        full NUM_SYMB_PER_SLOT symbols/slot - the default here). ekf.py's streaming-drift script
        embeds DMRS in-slot instead (see its module docstring), so only a fixed subset of each
        slot's symbols are LDPC-coded payload; its callers (ekf_predict_update,
        _log_static_syndrome_stats) pass that payload-symbols-per-slot count explicitly so this
        method's ldpc_n matches the codeword length ekf.py's own
        LDPC5GCodec actually used to encode that payload - otherwise the SyndromeLoss's parity
        checks are built for a different code than what's on the wire, making mean_p/mean_hard_sat
        structurally meaningless (see ekf-mod-pilot-syndrome-bug memory for a prior instance of
        this same class of bug). tag is purely cosmetic (the log-line prefix on the cached
        SyndromeLoss - see its own tag docstring) and reflects whichever caller happens to trigger
        construction first, since the instance is shared by both _syndrome_component (tsyn
        training loss) and ekf_predict_update (EKF)."""
        if int(getattr(conf, 'mcs', -1)) <= -1:
            return None
        qm, code_rate = get_mcs(conf.mcs)
        ldpc_n = int(conf.num_res * payload_symbols_per_slot * int(qm))
        ldpc_k = int(ldpc_n * code_rate)
        crc_length = 24 if ldpc_k > 3824 else 16
        key = (ldpc_k + crc_length, ldpc_n)
        if getattr(self, '_synd_key', None) != key:
            from python_code.coding.syndrome_loss import SyndromeLoss
            self._synd = SyndromeLoss(
                k=key[0], n=key[1], device=DEVICE,
                fallback_iters=int(getattr(conf, 'tsyn_fallback_iters', 0)), tag=tag)
            self._synd_key = key
            self._synd_qm = int(qm)
        return self._synd

    def _syndrome_component(self, est_rs: torch.Tensor, payload_symbols_per_slot: int = NUM_SYMB_PER_SLOT):
        """L_synd on the batch grid, tapped in decoder-input stream order.

        The per-user LLR grid (batch, num_bits, num_res) flattened in natural
        order equals the per-user LDPC decoder input stream that evaluate.py
        builds (symbol-major, then bit-in-symbol, then RE), so consecutive
        ldpc_n-bit windows are codewords — provided the batch preserves symbol
        order and spans whole slots. Returns (loss, hard_satisfaction) or None
        when the syndrome term cannot be formed for this batch.

        payload_symbols_per_slot: how many of the batch's rows make up one slot/codeword window -
        defaults to a full NUM_SYMB_PER_SLOT (evaluate.py's own pilot region, unchanged from
        before this parameter existed); ekf.py's callers pass their own DMRS-stripped
        payload-only count instead, so this must match whatever _get_syndrome_helper built its
        ldpc_n from (see that method's docstring) or the batch-alignment check below is checking
        the wrong unit.
        """
        helper = self._get_syndrome_helper(payload_symbols_per_slot=payload_symbols_per_slot)
        if helper is None:
            self._tsyn_warn_once('mcs', "training_loss='tsyn' needs conf.mcs > -1 (LDPC); "
                                        "syndrome term disabled, falling back to pure tent.")
            return None
        if not getattr(conf, 'encode_pilots', False) or conf.make_64QAM_16QAM_percentage != 0:
            self._tsyn_warn_once('encode_pilots', "pilot region is not LDPC-coded (needs "
                                 "encode_pilots: True and make_64QAM_16QAM_percentage: 0); "
                                 "L_synd would measure nonexistent code structure. "
                                 "Syndrome term disabled, falling back to pure tent.")
            return None
        if getattr(conf, 'shuffle', False) or getattr(conf, 'shuffle_augment_priors', False):
            self._tsyn_warn_once('shuffle', "shuffle/shuffle_augment_priors scramble symbol order, "
                                            "breaking codeword alignment; syndrome term disabled.")
            return None
        bs = int(getattr(conf, 'batch_size', 0))
        if bs > 0 and bs % payload_symbols_per_slot != 0:
            self._tsyn_warn_once('batch_align', f"batch_size={bs} is not a multiple of "
                                 f"payload_symbols_per_slot={payload_symbols_per_slot}, so mini-batches start "
                                 f"mid-codeword; syndrome term disabled. Use batch_size <= 0 "
                                 f"(full batch) or a multiple of {payload_symbols_per_slot}.")
            return None
        if est_rs.dim() != 3 or est_rs.shape[1] != self._synd_qm:
            self._tsyn_warn_once('shape', f"LLR grid shape {tuple(est_rs.shape)} does not match "
                                          f"qm={self._synd_qm} bits/symbol (first_half or pilot "
                                          f"modulation mismatch); syndrome term disabled.")
            return None
        stream = est_rs.reshape(-1)
        num_slots = int(stream.numel() // helper.n)
        if num_slots == 0:
            self._tsyn_warn_once('slots', f"batch too small for one codeword "
                                          f"({stream.numel()} < {helper.n} bits); syndrome term "
                                          f"disabled. Use batch_size <= 0 or a multiple of "
                                          f"{payload_symbols_per_slot} symbols.")
            return None
        slots = stream[:num_slots * helper.n].reshape(num_slots, helper.n)
        if getattr(self, 'synd_use_bp', False):    # ekf.py weights_track_mode='sgdsbp'
            return self._syndrome_component_bp(helper, slots)
        l_synd = helper.loss(slots)
        sat = helper.hard_satisfaction(slots.detach())
        return l_synd, sat

    def _syndrome_component_bp(self, helper, slots: torch.Tensor):
        """L_synd with ekfbp's measurement (ekf.py weights_track_mode='sgdsbp'): ALL mother-code check
        rows; each punctured column v in row m takes BP's extrinsic value tanh(L_{v->m}/2) from a detached
        sum-product BP (_ladder_bp, conf.bp_iters iterations) run on the current LLRs of that codeword -
        a constant, no gradient. Transmitted columns keep the network's tanh(L/2), the only part with a
        gradient. Same per-row penalty as SyndromeLoss.loss: -log((1 + p_m) / 2), averaged over rows and
        codewords. slots: (B, n) project-convention LLRs (L > 0 <=> bit 1). Returns (loss, sat) with
        sat = fraction of rows with p_m > 0 (the same soft-sign health metric ekf's mean_hard_sat uses)."""
        helper._to(slots.device)
        if getattr(self, '_sbp_helper', None) is not helper:
            _is_p = torch.zeros(helper.n_ldpc, dtype=torch.bool, device=helper.edge_bit_all.device)
            _is_p[helper.punctured_idx] = True
            self._sbp_punc_edge = _is_p[helper.edge_bit_all]
            self._sbp_helper = helper
        clamp = helper.LLR_CLAMP
        t_col = torch.tanh(helper.map_to_mother(slots).clamp(-clamp, clamp) / 2.0)   # (B, n_ldpc)
        rows = []
        for b in range(slots.shape[0]):
            v2c, _ = self._ladder_bp(helper, slots[b].detach())
            tpe = torch.tanh(v2c.clamp(-clamp, clamp) / 2.0)
            te = torch.where(self._sbp_punc_edge, tpe, t_col[b, helper.edge_bit_all])
            rows.append(self._ladder_row_products(te, helper.edge_check_all, helper.num_checks))
        p = torch.stack(rows)                                                           # (B, M)
        l_synd = -torch.log(((1.0 + p) / 2.0).clamp(min=1e-9, max=1.0)).mean()
        return l_synd, float((p.detach() > 0).float().mean())

    @staticmethod
    def _preprocess(rx: torch.Tensor) -> torch.Tensor:
        return rx.float()

    def _forward(self, rx: torch.Tensor, num_bits: int, n_users: int, iterations: int, probs_in: torch.Tensor) -> tuple[List, List]:
        # detect and decode
        detected_word_list = [None] * iterations
        llrs_mat_list = [None] * iterations
        if conf.which_augment == 'NO_AUGMENT':
            probs_vec = self._initialize_probs_for_infer(rx, num_bits, n_users)
        else:
            probs_vec = probs_in.to(DEVICE)

        nns = 0
        for i in range(iterations):
            probs_vec, llrs_mat_list[i] = self._calculate_posteriors(self.detector, i + 1, rx.to(device=DEVICE).unsqueeze(-1), probs_vec, num_bits, n_users, nns)
            detected_word_list[i] = self._compute_output(probs_vec)
        # plt.imshow(self.detector[0][0].fc1.weight[0, :, 0, :].cpu().detach(), cmap='gray')
        # pass

        return detected_word_list, llrs_mat_list



    def _compute_output(self, probs_vec):
        symbols_word = prob_to_BPSK_symbol(probs_vec.float())
        detected_word = BPSKModulator.demodulate(symbols_word)
        return detected_word

    def _prepare_data_for_training(self, tx: torch.Tensor, rx: torch.Tensor, probs_vec: torch.Tensor, n_users: int) -> [torch.Tensor, torch.Tensor]:
        """
        Generates the data for each user
        """
        tx_all = []
        rx_prob_all = []
        no_samples = getattr(conf, 'no_samples', False)
        for user in range(n_users):
            if conf.no_probs:
                rx_prob_all.append(rx.unsqueeze(-1))
            elif no_samples:
                rx_prob_all.append(probs_vec)
            else:
                rx_prob_all.append(torch.cat((rx.unsqueeze(-1), probs_vec), dim=1))
            tx_all.append(tx[:, user, :])
        return tx_all, rx_prob_all

    def _initialize_probs_for_training(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)

    def _initialize_probs(self, tx, num_bits, n_users):
        dim0 = int(tx.shape[0]//num_bits)
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        # rnd_init = torch.from_numpy(np.random.choice([0, 1], size=(dim0,dim1,dim2,dim3)).astype(np.float32))
        rnd_init = HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32)
        return rnd_init

    def _calculate_posteriors(self, model: List[List[nn.Module]], i: int, rx_real: torch.Tensor, prob: torch.tensor, num_bits: int, n_users: int, nns: torch.Tensor) -> torch.Tensor:
        """
        Propagates the probabilities through the learnt networks.
        """
        next_probs_vec = prob.clone()
        # next_probs_vec = torch.zeros_like(prob)
        llrs_mat = torch.zeros(next_probs_vec.shape).to(DEVICE)
        no_samples = getattr(conf, 'no_samples', False)
        for user in range(n_users):
            if conf.no_probs:
                rx_prob = rx_real
            elif no_samples:
                rx_prob = prob
            else:
                rx_prob = torch.cat((rx_real, prob), dim=1)

            with torch.no_grad():
                output, llrs = model[user][i - 1](rx_prob)
            index_start = user * num_bits
            index_end = (user + 1) * num_bits
            next_probs_vec[:, index_start:index_end, :, :] = output
            llrs_mat[:, index_start:index_end, :, :] = llrs

        return next_probs_vec, llrs_mat

    def _initialize_probs_for_infer(self, rx: torch.Tensor, num_bits: int, n_users: int):
        dim0 = rx.shape[0]
        dim1 = num_bits * n_users
        dim2 = conf.num_res
        dim3 = 1
        return HALF * torch.ones(dim0,dim1,dim2,dim3, dtype=torch.float32).to(DEVICE)

    def save_weights(self, path: str, extra_state: dict = None):
        """Save the full state_dict of every (user, iteration) ESCNN network to a single .pt
        file. extra_state (optional) is merged in under its own top-level string keys (e.g.
        'deepsic'/'deeprx') alongside ESCNN's own int-keyed {user: {iter: state_dict}} entries -
        int vs str keys can't collide, so this stays backward compatible with load_weights and
        with checkpoints saved before extra_state existed."""
        state = {user: {i: net.state_dict() for i, net in enumerate(nets)} for user, nets in enumerate(self.detector)}
        if extra_state:
            state.update(extra_state)
        torch.save(state, path)

    def load_weights(self, path: str):
        """Load weights previously written by save_weights into the current (user, iteration) networks."""
        state = torch.load(path, map_location=DEVICE, weights_only=True)
        for user, nets in enumerate(self.detector):
            for i, net in enumerate(nets):
                net.load_state_dict(state[user][i])

    def set_load_freeze(self, mode: str):
        """Apply set_load_freeze(mode) to every (user, iteration) ESCNN network."""
        for nets in self.detector:
            for net in nets:
                net.set_load_freeze(mode)