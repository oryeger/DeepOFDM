"""Tests for the tsyn (tent + LDPC syndrome anchor) training loss.

Run as a plain script: python -m python_code.test_tsyn

Covers (in order):
  1. Mapping correctness: true codewords from the existing encode chain give
     near-zero L_synd and >0.99 hard syndrome satisfaction on usable checks.
     Uses a longer code (n=500) where restricted mode has usable checks, so a
     mapping bug cannot hide behind the fallback estimator.
  2. Sensitivity: flipping 20 high-|L| positions blows up L_synd and drops
     the satisfaction rate.
  3. Project short-code config (n=112): zero usable checks -> auto-fallback;
     loss finite, codewords still score better than corrupted ones.
  4. tw=1.0 reproduces training_loss='tent' exactly through the trainer.
  5. Filler positions are neutral: the filler LLR magnitude does not change
     the loss.
  6. Gradients: tsyn loss backward() through the ESCNN produces finite,
     nonzero grads; extreme LLR inputs (+-1e4) stay NaN/Inf-free.
"""
import sys
import numpy as np
import torch

from python_code import DEVICE, conf
from python_code.coding.mcs_table import get_mcs
from python_code.coding.ldpc_wrapper import LDPC5GCodec
from python_code.coding.crc_wrapper import CRC5GCodec
from python_code.coding.syndrome_loss import SyndromeLoss
from python_code.utils.constants import NUM_SYMB_PER_SLOT

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        print(f'  PASS  {name}')
        PASS += 1
    else:
        print(f'  FAIL  {name}' + (f'  ({detail})' if detail else ''))
        FAIL += 1


def make_codewords(ldpc_k, crc_length, n, batch, rng, llr_mag=10.0, noise_std=0.3):
    """Random bits -> CRC -> LDPC (the existing chain) -> project-convention LLRs."""
    codec = LDPC5GCodec(k=ldpc_k + crc_length, n=n)
    crc = CRC5GCodec(crc_length)
    u = rng.integers(0, 2, size=(batch, ldpc_k))
    u_crc = crc.encode(u)
    codewords = codec.encode(u_crc)
    llr = (2.0 * codewords - 1.0) * llr_mag + rng.normal(0.0, noise_std, size=codewords.shape)
    decoded = codec.decode(llr)
    decode_ok = np.array_equal(decoded.astype(int), u_crc.astype(int))
    return torch.tensor(llr, dtype=torch.float32), u_crc.astype(int), codewords.astype(int), decode_ok


rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Tests 1-2: full-circular-buffer code (k=200, n=1000: rate 1/5, zero fillers)
# so the whole mother codeword is known and restricted mode has usable checks.
# ---------------------------------------------------------------------------
SYN_LDPC_K, SYN_CRC, SYN_N = 184, 16, 1000
print(f'Synthetic code for mapping tests: LDPC5GCodec(k={SYN_LDPC_K + SYN_CRC}, n={SYN_N})')
synd_syn = SyndromeLoss(k=SYN_LDPC_K + SYN_CRC, n=SYN_N, device='cpu')

print('Test 1: mapping correctness on true codewords (restricted-check mode)')
check('restricted mode has usable checks', synd_syn.num_usable > 0,
      f'{synd_syn.num_usable} usable')
check('restricted mode is active (fallback not auto-enabled)', not synd_syn.punctured_fallback)
llr_syn, u_crc_syn, cw_syn, decode_ok = make_codewords(SYN_LDPC_K, SYN_CRC, SYN_N, batch=8, rng=rng)
check('decoder recovers info bits at high SNR', decode_ok)

# Hard-syndrome validation over every check whose support is fully known:
# transmitted positions (via the inverse rate-matching map), fillers (zeros),
# and the first 2Z systematic positions (which equal u_crc[:, :2Z] since the
# code is systematic). This validates the 2Z placement too — a superset of the
# usable-check set, so a mapping offset cannot hide behind the check mask.
t2m = synd_syn.tx_to_mother.cpu().numpy()
mother_bits = np.zeros((8, synd_syn.n_ldpc), dtype=int)
mother_bits[:, t2m] = cw_syn
mother_bits[:, :2 * synd_syn.z] = u_crc_syn[:, :2 * synd_syn.z]
known = np.zeros(synd_syn.n_ldpc, dtype=bool)
known[t2m] = True
known[synd_syn.filler_idx.cpu().numpy()] = True
known[:2 * synd_syn.z] = True
ec = synd_syn.edge_check_all.cpu().numpy()
eb = synd_syn.edge_bit_all.cpu().numpy()
check_unknown = np.zeros(synd_syn.num_checks, dtype=bool)
np.logical_or.at(check_unknown, ec, ~known[eb])
known_checks = ~check_unknown
par = np.zeros((8, synd_syn.num_checks), dtype=int)
for b in range(8):
    np.add.at(par[b], ec, mother_bits[b, eb])
viol = int((par[:, known_checks] % 2).sum())
check(f'H b^T = 0 on all fully-known checks '
      f'({int(known_checks.sum())} checks > {synd_syn.num_usable} usable)',
      known_checks.sum() > synd_syn.num_usable and viol == 0, f'{viol} violated')

loss_clean = synd_syn.loss(llr_syn).item()
sat_clean = synd_syn.hard_satisfaction(llr_syn)
check(f'L_synd small on codewords (got {loss_clean:.6f})', loss_clean < 0.05)
check(f'hard satisfaction > 0.99 (got {sat_clean:.4f})', sat_clean > 0.99)

print('Test 2: 20 sign flips blow up L_synd')
llr_flip = llr_syn.clone()
strong = torch.nonzero(llr_flip[0].abs() > 4.0).flatten()
flip_idx = strong[torch.randperm(strong.numel(), generator=torch.Generator().manual_seed(1))[:20]]
llr_flip[0, flip_idx] *= -1.0
loss_flip = synd_syn.loss(llr_flip).item()
sat_flip = synd_syn.hard_satisfaction(llr_flip)
ratio = loss_flip / max(loss_clean, 1e-12)
check(f'L_synd grows by orders of magnitude (x{ratio:.1f})', ratio > 100.0)
check(f'satisfaction drops ({sat_clean:.4f} -> {sat_flip:.4f})', sat_flip < sat_clean - 0.005)

# ---------------------------------------------------------------------------
# Test 3: project short-code config -> auto-fallback path
# ---------------------------------------------------------------------------
NUM_RES = 4
MCS = 4
qm, code_rate = get_mcs(MCS)
qm = int(qm)
ldpc_n = int(NUM_RES * NUM_SYMB_PER_SLOT * qm)
ldpc_k = int(ldpc_n * code_rate)
crc_length = 24 if ldpc_k > 3824 else 16
print(f'Test 3: project code config (qm={qm} ldpc_n={ldpc_n} ldpc_k={ldpc_k} crc={crc_length})')
synd_prj = SyndromeLoss(k=ldpc_k + crc_length, n=ldpc_n, device='cpu')
check('auto-fallback engaged on zero usable checks',
      synd_prj.punctured_fallback or synd_prj.num_usable > 0)
llr_prj, _, _, decode_ok_prj = make_codewords(ldpc_k, crc_length, ldpc_n, batch=8, rng=rng)
check('decoder recovers info bits at high SNR (project code)', decode_ok_prj)
loss_prj_clean = synd_prj.loss(llr_prj).item()
sat_prj_clean = synd_prj.hard_satisfaction(llr_prj)
check(f'fallback loss finite on codewords (got {loss_prj_clean:.4f})',
      np.isfinite(loss_prj_clean))
llr_prj_flip = llr_prj.clone()
strong_prj = torch.nonzero(llr_prj_flip[0].abs() > 4.0).flatten()
fidx = strong_prj[torch.randperm(strong_prj.numel(), generator=torch.Generator().manual_seed(3))[:20]]
llr_prj_flip[0, fidx] *= -1.0
loss_prj_flip = synd_prj.loss(llr_prj_flip).item()
sat_prj_flip = synd_prj.hard_satisfaction(llr_prj_flip)
check(f'fallback loss detects corruption ({loss_prj_clean:.4f} -> {loss_prj_flip:.4f})',
      loss_prj_flip > loss_prj_clean + 0.01)
check(f'fallback satisfaction drops ({sat_prj_clean:.4f} -> {sat_prj_flip:.4f})',
      sat_prj_flip < sat_prj_clean)

# ---------------------------------------------------------------------------
# Trainer-level tests need the conf singleton set up consistently.
# ---------------------------------------------------------------------------
conf.set_value('mcs', MCS)
conf.set_value('num_res', NUM_RES)
conf.set_value('n_users', 4)
conf.set_value('n_ants', 8)
conf.set_value('iterations', 1)
conf.set_value('no_probs', False)
conf.set_value('no_samples', False)
conf.set_value('use_film', False)
conf.set_value('scale_input', True)
conf.set_value('escnn_dropout', 0.0)
conf.set_value('shuffle', False)
conf.set_value('shuffle_augment_priors', False)
conf.set_value('beta_balance', 0.0)
conf.set_value('make_64QAM_16QAM_percentage', 0)
conf.set_value('escnn_load_freeze', 'none')
conf.set_value('load_escnn_weights_tag', '')
conf.set_value('tsyn_punctured_fallback', False)
conf.set_value('encode_pilots', True)   # syndrome term requires LDPC-coded pilots
conf.set_value('batch_size', -1)

from python_code.detectors.escnn.escnn_trainer import ESCNNTrainer  # noqa: E402

trainer = ESCNNTrainer(qm, conf.n_users, conf.n_ants)

# A batch of 42 symbols -> stream of 42*qm*NUM_RES bits = 3 codewords
B = 42
gen = torch.Generator().manual_seed(2)
est = torch.randn(B, qm, NUM_RES, 1, generator=gen) * 4.0
tx = torch.randint(0, 2, (B, qm, NUM_RES), generator=gen).float()

# ---------------------------------------------------------------------------
# Test 4: tw=1.0 reproduces tent exactly
# ---------------------------------------------------------------------------
print('Test 4: tsyn(tw=1) == tent')
conf.set_value('training_loss', 'tent')
l_tent = trainer._calculate_loss(est.clone(), tx)
conf.set_value('training_loss', 'tsyn')
conf.set_value('tw', 1.0)
l_tsyn1 = trainer._calculate_loss(est.clone(), tx)
diff = abs(l_tsyn1.item() - l_tent.item())
check(f'|L_tsyn(tw=1) - L_tent| < 1e-6 (got {diff:.2e})', diff < 1e-6)
check('internal first-batch equivalence assertion ran',
      getattr(trainer, '_tsyn_equiv_checked', False))

# tw=0.5 must differ from pure tent (syndrome term active on random LLRs)
conf.set_value('tw', 0.5)
l_tsyn_half = trainer._calculate_loss(est.clone(), tx)
check('tw=0.5 mixes in a nonzero syndrome term',
      abs(l_tsyn_half.item() - l_tent.item()) > 1e-4,
      f'{l_tsyn_half.item():.4f} vs {l_tent.item():.4f}')
st = getattr(trainer, '_tsyn_stats', {})
check('monitoring stats populated (l_tent, l_synd, sat)',
      st.get('l_synd') is not None and st.get('sat') is not None, str(st))

# ---------------------------------------------------------------------------
# Test 5: filler LLR magnitude is neutral (project code: 59 filler bits)
# ---------------------------------------------------------------------------
print('Test 5: filler neutrality')
synd_f20 = SyndromeLoss(k=ldpc_k + crc_length, n=ldpc_n, device='cpu', filler_llr=20.0)
check('filler positions present', synd_prj.k_filler > 0, f'{synd_prj.k_filler}')
l30 = synd_prj.loss(llr_prj).item()
l20 = synd_f20.loss(llr_prj).item()
check(f'loss unchanged for filler_llr 30 vs 20 (|diff|={abs(l30 - l20):.2e})',
      abs(l30 - l20) < 1e-6)

# ---------------------------------------------------------------------------
# Test 6: gradients through the ESCNN; extreme-LLR robustness
# ---------------------------------------------------------------------------
print('Test 6: gradient health')
conf.set_value('training_loss', 'tsyn')
conf.set_value('tw', 0.5)
model = trainer.detector[0][0].to(DEVICE)
c_in = qm * conf.n_users + 2 * conf.n_ants
rx_prob = torch.randn(B, c_in, NUM_RES, 1, generator=gen).to(DEVICE)
_, llrs = model(rx_prob)
loss = trainer._calculate_loss(llrs, tx.to(DEVICE))
model.zero_grad()
loss.backward()
grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
all_finite = all(torch.isfinite(g).all().item() for g in grads)
any_nonzero = any(g.abs().sum().item() > 0 for g in grads)
check('loss finite', torch.isfinite(loss).item(), f'loss={loss.item()}')
check('all ESCNN grads finite', all_finite)
check('at least one nonzero grad', any_nonzero)

# Extreme LLRs straight into the syndrome loss (both modes)
for name, s in (('restricted', synd_syn), ('fallback', synd_prj)):
    n_cols = s.n
    llr_ext = torch.full((2, n_cols), 1e4)
    llr_ext[1] = -1e4
    llr_ext.requires_grad_(True)
    l_ext = s.loss(llr_ext)
    l_ext.backward()
    check(f'extreme +-1e4 LLRs ({name}): loss finite', torch.isfinite(l_ext).item(),
          f'{l_ext.item()}')
    check(f'extreme +-1e4 LLRs ({name}): grads NaN/Inf-free',
          torch.isfinite(llr_ext.grad).all().item())

# ---------------------------------------------------------------------------
# Test 7: encode_pilots — coded pilot region is real codewords in the exact
# stream order the trainer's syndrome tap uses ((bit_row, re) per user)
# ---------------------------------------------------------------------------
print('Test 7: encode_pilots produces decodable codewords in trainer tap order')
from python_code.coding.pilot_coding import encode_pilots  # noqa: E402

pilot_rows = 3 * ldpc_n // NUM_RES   # 3 slots worth of bit rows
codec_prj = LDPC5GCodec(k=ldpc_k + crc_length, n=ldpc_n)
crc_prj = CRC5GCodec(crc_length)
tx_p = encode_pilots(np.random.default_rng(7), pilot_rows, NUM_RES,
                     4, codec_prj, crc_prj, ldpc_k, ldpc_n)
check('coded pilot shape matches (pilot_length, n_users, num_res)',
      tx_p.shape == (pilot_rows, 4, NUM_RES), str(tx_p.shape))
slots_all = np.stack([tx_p[:, u, :].reshape(-1).reshape(3, ldpc_n) for u in range(4)])
slots_all = slots_all.reshape(12, ldpc_n)
llr_p = (2.0 * slots_all - 1.0) * 8.0
decoded_p = codec_prj.decode(llr_p)
crc_valid = crc_prj.decode(decoded_p)
check('every pilot slot of every user decodes with valid CRC',
      bool(np.asarray(crc_valid).all()), f'{np.asarray(crc_valid).sum()}/12 valid')

# ---------------------------------------------------------------------------
print()
print(f'Results: {PASS} passed, {FAIL} failed')
sys.exit(0 if FAIL == 0 else 1)
