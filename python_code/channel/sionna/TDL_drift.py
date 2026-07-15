"""Channel-drift snapshot generation for slow-time-varying-channel experiments.

conf.channel_drift_index selects a point in time (in OFDM slots) along the
TDL channel trajectory implied by (channel_seed, channel_model, speed):
  channel_drift_index=0   -> identical to the ordinary (non-drift) channel
  channel_drift_index=N   -> the channel as it would be N slots later,
                              deterministic given the same channel_seed/speed

This works by evaluating the TDL model's closed-form sinusoid sum directly
at an arbitrary time offset, without generating (or discarding) the
intervening samples: the per-path Doppler/angle/phase parameters are drawn
once (fixed size, independent of how far into the trajectory we look), and
`h(t)` is then a deterministic function of t alone. See TDL.__call__ in the
Sionna source for the reference implementation this mirrors; call_offset()
below is line-for-line identical except sample_times carries an added
offset instead of always starting at 0.

Verified two ways:
  1. Runtime: local Sionna 1.0.2 (exact match to a vanilla TDL() call at
     offset=0, O(1) cost regardless of offset magnitude, deterministic
     given the same seed).
  2. Source comparison: Sionna 0.19.2 (the version on the cluster, per
     sionna.channel.tr38901.TDL.__call__) uses an essentially identical
     formula/draw order/shapes to 1.0.2 - including the same literal `1`
     (not num_clusters) for doppler's cluster-axis shape, which was the
     actual bug caught during development of this module. The one textual
     difference (0.19.2 omits adjoint_b=True on the tx_corr_mat matmul) is
     inert for this codebase, since tx_corr is always the identity matrix
     here (see TLD_channel.py's _get_spatial_correlation_matrices).
_self_check() below still runs on every process as a live guard, in case a
future Sionna upgrade changes this again.
"""
import numpy as np
import tensorflow as tf

try:
    from sionna.channel.tr38901 import TDL
    from sionna.channel import cir_to_time_channel, time_lag_discrete_time_channel
    from sionna.utils import insert_dims, flatten_last_dims, split_dim
    import sionna as _sn
    _config = _sn.config
except ImportError:  # Sionna >= 1.0 moved channel/config under sionna.phy
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel import cir_to_time_channel, time_lag_discrete_time_channel
    from sionna.phy.utils import insert_dims, flatten_last_dims, split_dim
    from sionna.phy import config as _config

PI = np.pi

_self_check_passed = False


class _TDLOffset(TDL):
    """TDL subclass that evaluates the same stochastic process at an
    arbitrary time offset. See module docstring for why this is valid."""

    def call_offset(self, batch_size, num_time_steps, sampling_frequency, time_offset_samples):
        # Attribute name differs across Sionna versions: 1.x exposes the
        # public property `rdtype`; 0.19.2 (cluster) only has the private
        # `_real_dtype`. Resolve once so the rest of this method is
        # version-agnostic.
        rdtype = getattr(self, 'rdtype', None)
        if rdtype is None:
            rdtype = self._real_dtype

        sample_times = (tf.range(num_time_steps, dtype=rdtype)
                         + tf.cast(time_offset_samples, rdtype)) / sampling_frequency
        sample_times = tf.expand_dims(insert_dims(sample_times, 6, 0), -1)

        # NOTE: shapes below (in particular the literal `1` for the
        # "clusters" axis of `doppler`) must match TDL.__call__ exactly -
        # a mismatch here desynchronizes every subsequent random draw
        # without raising an error. Verified via the self-check below.
        doppler = _config.tf_rng.uniform([batch_size, 1, 1, 1, 1, 1, 1, 1],
                                          self._min_doppler, self._max_doppler, rdtype)
        theta = _config.tf_rng.uniform(
            [batch_size, 1, 1, 1, 1, self._num_clusters, 1, self._num_sinusoids],
            -PI / tf.cast(self._num_sinusoids, rdtype),
            PI / tf.cast(self._num_sinusoids, rdtype), rdtype)
        alpha = self._alpha_const + theta
        phi = _config.tf_rng.uniform(
            [batch_size, 1, self._num_rx_ant, 1, self._num_tx_ant, self._num_clusters, 1, self._num_sinusoids],
            -PI, PI, rdtype)

        argument = doppler * sample_times * tf.cos(alpha) + phi
        h = tf.complex(tf.cos(argument), tf.sin(argument))
        normalization_factor = 1. / tf.sqrt(tf.cast(self._num_sinusoids, rdtype))
        h = tf.complex(normalization_factor, tf.constant(0., rdtype)) * tf.reduce_sum(h, axis=-1)
        mean_powers = tf.expand_dims(insert_dims(self._mean_powers, 5, 0), -1)
        h = tf.sqrt(mean_powers) * h

        if self._los:
            phi_0 = _config.tf_rng.uniform([batch_size, 1, 1, 1, 1, 1, 1], -PI, PI, rdtype)
            doppler2 = tf.squeeze(doppler, axis=-1)
            sample_times2 = tf.squeeze(sample_times, axis=-1)
            arg_spec = doppler2 * sample_times2 * tf.cos(self._los_angle_of_arrival) + phi_0
            h_spec = tf.complex(tf.cos(arg_spec), tf.sin(arg_spec))
            h = tf.concat([h_spec * tf.sqrt(self._los_power) + h[:, :, :, :, :, :1, :],
                           h[:, :, :, :, :, 1:, :]], axis=5)

        if self._scale_delays:
            delays = self._delays * self._delay_spread
        else:
            delays = self._delays * 1e-9
        delays = insert_dims(delays, 3, 0)
        delays = tf.tile(delays, [batch_size, 1, 1, 1])

        if self._spatial_corr_mat_sqrt is not None:
            h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
            h = flatten_last_dims(h, 2)
            h = tf.expand_dims(h, axis=-1)
            h = tf.matmul(self._spatial_corr_mat_sqrt, h)
            h = tf.squeeze(h, axis=-1)
            h = split_dim(h, [self._num_rx_ant, self._num_tx_ant], tf.rank(h) - 1)
            h = tf.transpose(h, [0, 1, 5, 2, 6, 3, 4])
        else:
            if (self._rx_corr_mat_sqrt is not None) or (self._tx_corr_mat_sqrt is not None):
                h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
                if self._rx_corr_mat_sqrt is not None:
                    h = tf.matmul(self._rx_corr_mat_sqrt, h)
                if self._tx_corr_mat_sqrt is not None:
                    h = tf.matmul(h, self._tx_corr_mat_sqrt, adjoint_b=True)
                h = tf.transpose(h, [0, 1, 5, 2, 6, 3, 4])

        h = tf.stop_gradient(h)
        delays = tf.stop_gradient(delays)
        return h, delays


def _self_check(tdl_kwargs, num_time_steps, sampling_frequency, seed):
    """Asserts call_offset(offset=0) reproduces a vanilla TDL() call exactly.
    Runs once per process. Raises loudly on mismatch (e.g. a Sionna version
    with different internals) instead of silently returning wrong channels.
    """
    global _self_check_passed
    if _self_check_passed:
        return
    _config.seed = seed
    tdl_vanilla = TDL(**tdl_kwargs)
    cir_vanilla = tdl_vanilla(1, num_time_steps, sampling_frequency)

    _config.seed = seed
    tdl_offset = _TDLOffset(**tdl_kwargs)
    cir_offset0 = tdl_offset.call_offset(1, num_time_steps, sampling_frequency, 0)

    if not np.allclose(cir_vanilla[0].numpy(), cir_offset0[0].numpy()):
        raise RuntimeError(
            "TDL_drift self-check failed: offset=0 does not reproduce a vanilla "
            "TDL() call with this Sionna installation. The internal random-draw "
            "order/shapes assumed by _TDLOffset.call_offset likely differ on "
            "this Sionna version - channel_drift_index cannot be trusted here "
            "without updating _TDLOffset to match this version's TDL.__call__.")
    _self_check_passed = True


def generate_drift_channel(tdl_kwargs: dict, num_time_samples: int, sampling_frequency: float,
                            seed: int, channel_drift_index: int, num_samples_per_slot: int,
                            batch_size: int = 1):
    """Returns h_time for the channel at channel_drift_index slots into the
    trajectory implied by (seed, tdl_kwargs['model'], tdl_kwargs['min_speed']).
    channel_drift_index=0 is identical to a fresh, ordinary TDL generation.
    """
    l_min, l_max = time_lag_discrete_time_channel(sampling_frequency)
    l_tot = l_max - l_min + 1
    _self_check(tdl_kwargs, num_time_samples, sampling_frequency, seed)

    _config.seed = seed
    tdl = _TDLOffset(**tdl_kwargs)
    time_offset_samples = int(channel_drift_index) * num_samples_per_slot
    cir = tdl.call_offset(batch_size, num_time_samples, sampling_frequency, time_offset_samples)
    h_time = cir_to_time_channel(sampling_frequency, *cir, l_min, l_max, normalize=True)
    return h_time
