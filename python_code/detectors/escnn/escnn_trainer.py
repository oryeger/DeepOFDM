
import math
from datetime import datetime
from typing import List

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from python_code import DEVICE, conf
from python_code.channel.modulator import BPSKModulator
from python_code.coding.mcs_table import get_mcs
from python_code.detectors.escnn.escnn_detector import ESCNNDetector
from python_code.detectors.trainer import Trainer
from python_code.utils.constants import HALF, TRAIN_PERCENTAGE, NUM_SYMB_PER_SLOT
from python_code.utils.probs_utils import prob_to_BPSK_symbol
from python_code.utils.constants import SHOW_ALL_ITERATIONS
from python_code.utils.probs_utils import ensure_tensor_iterable
from python_code.utils.probs_utils import pilot_third_bit_mask

Softmax = torch.nn.Softmax(dim=1)

class ESCNNTrainer(Trainer):

    def __init__(self, num_bits: int, n_users: int, n_ants: int):
        super().__init__(num_bits, n_users, n_ants)

    def __str__(self):
        return 'ESCNN'

    def _deep_learning_setup(self, single_model):
        weight_decay = float(getattr(conf, 'escnn_weight_decay', 0.0))
        self.optimizer = Adam(filter(lambda p: p.requires_grad, single_model.parameters()),
                              lr=self.lr, weight_decay=weight_decay)
        from torch.nn import BCEWithLogitsLoss
        self.criterion = BCEWithLogitsLoss(reduction='none').to(DEVICE)

    def _initialize_detector(self, num_bits, n_users, n_ants):

        self.detector = [[ESCNNDetector(num_bits, n_users).to(DEVICE) for _ in range(conf.iterations)] for _ in
                         range(n_users)]  # 2D list for Storing the ESCNN Networks

    def _train_model(self, single_model: nn.Module, tx: torch.Tensor, rx_prob: torch.Tensor, num_bits:int, epochs: int, first_half_flag: bool, stage: str, network_id: str = "") -> list[float]:
        """
        Trains a ESCNN Network and returns the total training loss.
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

        # tsyn: codeword windows start at symbol 0, so every region the loss
        # sees must begin on a slot boundary (NUM_SYMB_PER_SLOT symbols).
        _slot_align = (getattr(conf, 'training_loss', 'bce') == 'tsyn')

        # Restrict to primary detector's validation portion only
        if getattr(conf, 'escnn_use_primary_val_only', False):
            primary_train_samples = rx_prob.shape[0] // 2
            if _slot_align:
                primary_train_samples -= primary_train_samples % NUM_SYMB_PER_SLOT
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

        # Split into train and validation sets
        train_samples = int(rx_prob.shape[0] * TRAIN_PERCENTAGE / 100)
        if _slot_align:
            train_samples -= train_samples % NUM_SYMB_PER_SLOT
        rx_prob_train = rx_prob[:train_samples]
        rx_prob_val = rx_prob[train_samples:]
        tx_train = tx_reshaped[:train_samples]
        tx_val = tx_reshaped[train_samples:]
        mask_train = loss_mask[:train_samples]
        mask_val = loss_mask[train_samples:]

        # Create DataLoader for mini-batch training
        # batch_size <= 0 means full-batch (no mini-batching)
        batch_size = conf.batch_size if hasattr(conf, 'batch_size') else 32
        if batch_size <= 0:
            batch_size = len(rx_prob_train)  # Full batch
        shuffle = conf.shuffle if hasattr(conf, 'shuffle') else True
        train_dataset = TensorDataset(rx_prob_train, tx_train, mask_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        es_patience = int(getattr(conf, 'early_stopping_patience', -1))
        es_enabled = es_patience > 0
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

                loss = self._calculate_loss(llrs_cur, batch_tx_cur, batch_mask_cur)
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

            # Calculate validation loss
            with torch.no_grad():
                rx_prob_val_device = rx_prob_val.to(DEVICE)
                _, llrs_val = single_model(rx_prob_val_device)
                if first_half_flag:
                    llrs_val_cur = llrs_val[:, 0::2, :, :]
                    mask_val_cur = mask_val[:, 0::2, :].to(DEVICE)
                    val_loss = self._calculate_loss(llrs_val_cur, tx_val[:, 0::2, :].to(DEVICE), mask_val_cur)
                else:
                    llrs_val_cur = llrs_val
                    mask_val_cur = mask_val.to(DEVICE)
                    val_loss = self._calculate_loss(llrs_val_cur, tx_val.to(DEVICE), mask_val_cur)
                val_loss_vect.append(val_loss.item())
                val_hb = self._calc_h_marginal(llrs_val_cur, mask_val_cur) if _log_hb else None
                val_tsyn = dict(getattr(self, '_tsyn_stats', {})) if _log_tsyn else None

            def _hb_suffix(train_hb, v_hb):
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
                            print(msg, flush=True)
                        break

            if log_enabled and (epoch + 1) % log_every == 0:
                msg = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} epoch {epoch+1}/{epochs} train={avg_train_loss:.4f} val={val_loss.item():.4f}"
                if es_enabled:
                    msg += f" best_val={best_val_loss:.4f}"
                if _log_hb:
                    msg += _hb_suffix(avg_train_hb, val_hb)
                if _log_tsyn:
                    msg += _tsyn_suffix()
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
                      rx_prob_all: List[torch.Tensor], num_bits: int, n_users: int, epochs: int, first_half_flag: bool, stage: str):
        train_loss_vect_user = []
        val_loss_vect_user = []
        for user in range(n_users):
            net_id = f"u={user} it={i}"
            train_loss_vect , val_loss_vect = self._train_model(model[user][i], tx_all[user], rx_prob_all[user].to(DEVICE), num_bits, epochs, first_half_flag, stage, network_id=net_id)
            if user == 0:
                train_loss_vect_user = train_loss_vect
                val_loss_vect_user = val_loss_vect
        return train_loss_vect_user , val_loss_vect_user




    def _online_training(self, tx: torch.Tensor, rx_real: torch.Tensor, num_bits: int, n_users: int, iterations: int, epochs: int, first_half_flag: bool, probs_in: torch.Tensor, stage: str = "base"):
        """
        Main training function for ESCNN trainer. Initializes the probabilities, then propagates them through the
        network, training sequentially each network and not by end-to-end manner (each one individually).
        """

        if conf.which_augment == 'NO_AUGMENT':
            initial_probs = self._initialize_probs(tx, num_bits, n_users)
        else:
            initial_probs = probs_in

        # Training the ESCNN network for each user for iteration=1
        tx_all, rx_prob_all = self._prepare_data_for_training(tx, rx_real, initial_probs, n_users)
        train_loss_vect , val_loss_vect = self._train_models(self.detector, 0, tx_all, rx_prob_all, num_bits, n_users, epochs, first_half_flag, stage)
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
            train_loss_cur , val_loss_cur =  self._train_models(self.detector, i, tx_all, rx_prob_all, num_bits, n_users, epochs, first_half_flag, stage)
            if SHOW_ALL_ITERATIONS:
                train_loss_vect = train_loss_vect + train_loss_cur
                val_loss_vect = val_loss_vect + val_loss_cur
        return train_loss_vect , val_loss_vect

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

    def _calculate_loss(self, est: torch.Tensor, tx: torch.IntTensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Loss dispatched by conf.training_loss:
          'bce'  - BCEWithLogitsLoss (sigmoid applied internally); uses tx and mask.
          'gfmi' - Blind EXIT-MI loss: minimises 1-GFMI = mean binary entropy of |L|.
                   Accepts but ignores tx (kept for signature compatibility).
                   Mask is still applied so constant pilot bits are excluded.
          'tsyn' - tw * L_tent + (1-tw) * L_synd, where L_synd is a soft LDPC
                   syndrome penalty over the batch's codeword LLRs.
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
            synd_out = self._syndrome_component(est_rs)
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

    def _tsyn_warn_once(self, key: str, msg: str):
        if not hasattr(self, '_tsyn_warned'):
            self._tsyn_warned = set()
        if key not in self._tsyn_warned:
            print(f"[tsyn] WARNING: {msg}", flush=True)
            self._tsyn_warned.add(key)

    def _get_syndrome_helper(self):
        """Lazily build/cache the SyndromeLoss for the current LDPC configuration
        (same k/n construction as evaluate.py's LDPC5GCodec)."""
        if int(getattr(conf, 'mcs', -1)) <= -1:
            return None
        qm, code_rate = get_mcs(conf.mcs)
        ldpc_n = int(conf.num_res * NUM_SYMB_PER_SLOT * int(qm))
        ldpc_k = int(ldpc_n * code_rate)
        crc_length = 24 if ldpc_k > 3824 else 16
        key = (ldpc_k + crc_length, ldpc_n)
        if getattr(self, '_synd_key', None) != key:
            from python_code.coding.syndrome_loss import SyndromeLoss
            self._synd = SyndromeLoss(
                k=key[0], n=key[1], device=DEVICE,
                punctured_fallback=bool(getattr(conf, 'tsyn_punctured_fallback', False)))
            self._synd_key = key
            self._synd_qm = int(qm)
        return self._synd

    def _syndrome_component(self, est_rs: torch.Tensor):
        """L_synd on the batch grid, tapped in decoder-input stream order.

        The per-user LLR grid (batch, num_bits, num_res) flattened in natural
        order equals the per-user LDPC decoder input stream that evaluate.py
        builds (symbol-major, then bit-in-symbol, then RE), so consecutive
        ldpc_n-bit windows are codewords — provided the batch preserves symbol
        order and spans whole slots. Returns (loss, hard_satisfaction) or None
        when the syndrome term cannot be formed for this batch.
        """
        helper = self._get_syndrome_helper()
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
        if bs > 0 and bs % NUM_SYMB_PER_SLOT != 0:
            self._tsyn_warn_once('batch_align', f"batch_size={bs} is not a multiple of "
                                 f"NUM_SYMB_PER_SLOT={NUM_SYMB_PER_SLOT}, so mini-batches start "
                                 f"mid-codeword; syndrome term disabled. Use batch_size <= 0 "
                                 f"(full batch) or a multiple of {NUM_SYMB_PER_SLOT}.")
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
                                          f"{NUM_SYMB_PER_SLOT} symbols.")
            return None
        slots = stream[:num_slots * helper.n].reshape(num_slots, helper.n)
        l_synd = helper.loss(slots)
        sat = helper.hard_satisfaction(slots.detach())
        return l_synd, sat

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

    def save_weights(self, path: str):
        """Save the full state_dict of every (user, iteration) ESCNN network to a single .pt file."""
        state = {user: {i: net.state_dict() for i, net in enumerate(nets)} for user, nets in enumerate(self.detector)}
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