"""Artificial Gaussian noise, measured against decoded PCM clean samples."""

import hashlib

import numpy as np


POWER_RULE = "mean(x**2) over ALL decoded mono PCM_16 samples, including silence; no VAD, centering or DC removal"
SNR_RULE = "10*log10(mean((gain*clean_pcm)**2)/mean((noisy_pcm-gain*clean_pcm)**2)); residual includes PCM_16 quantization"
SNR_TOLERANCE_DB = 0.05


def record_seed(seed, sample_id, voice, snr):
    key = f"{seed}:{sample_id}:{voice}:{snr:g}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def power(signal):
    return float(np.mean(np.asarray(signal, dtype=np.float64) ** 2))


def mix_noise(clean, snr_db, seed):
    clean = np.asarray(clean, dtype=np.float64)
    if clean.ndim != 1 or not clean.size or not np.isfinite(clean).all() or power(clean) <= 0:
        raise ValueError("Noise requires nonempty finite mono signal with positive power")
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    rng = np.random.Generator(np.random.PCG64(seed))
    noise = rng.standard_normal(len(clean))
    noise_scale = np.sqrt(power(clean) / (power(noise) * 10 ** (snr_db / 10)))
    mixture = clean + noise_scale * noise
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, 0.98 / peak)
    return gain * mixture, {
        "type": "additive_noise", "noise_type": "artificial_white_gaussian",
        "source": "synthetic RNG, not a real acoustic environment recording",
        "snr_db": snr_db, "seed": seed, "rng": "numpy.PCG64.standard_normal",
        "noise_scale": float(noise_scale), "mixture_gain": gain,
        "clean_power": power(clean), "raw_noise_power": power(noise),
        "pre_gain_peak": peak, "sample_count": len(clean),
        "power_rule": POWER_RULE, "snr_measurement_rule": SNR_RULE,
    }


def measured_snr(clean, noisy_pcm, gain):
    clean = np.asarray(clean, dtype=np.float64)
    noisy_pcm = np.asarray(noisy_pcm, dtype=np.float64)
    if clean.shape != noisy_pcm.shape:
        raise ValueError("Clean/noisy sample counts differ")
    signal = gain * clean
    residual_power = power(noisy_pcm - signal)
    if power(signal) <= 0 or residual_power <= 0:
        raise ValueError("SNR requires positive signal and residual power")
    return float(10 * np.log10(power(signal) / residual_power))
