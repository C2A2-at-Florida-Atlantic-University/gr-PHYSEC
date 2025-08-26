#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
from typing import Optional

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

# Read IQ samples from file
def read_iq(file_path: str):
    # Read IQ samples from file
    with open(file_path, 'rb') as f:
        # Read IQ samples from .iq file
        iq = np.fromfile(f, dtype=np.complex64)
    return iq

def summarize(samples: np.ndarray, samp_rate: Optional[float]) -> None:
    n = samples.size
    if n == 0:
        print("No samples loaded.")
        return
    pwr = np.mean(np.abs(samples) ** 2)
    print(f"Loaded complex samples: {n}")
    if samp_rate:
        dur = n / float(samp_rate)
        print(f"Sample rate: {samp_rate:.3f} Sa/s  (~{dur:.3f}s)")
    print(f"Mean power: {pwr:.6f}")
    print(f"First 5 samples: {samples[:5]}")


def plot_wave(samples: np.ndarray, samp_rate: Optional[float], num: int) -> None:
    if not HAVE_PLOT:
        print("matplotlib not available; skipping plots.")
        return
    if num is None or num <= 0:
        num = samples.size
    else:
        num = min(num, samples.size)
    t = np.arange(num)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(t, samples.real, label="I")
    ax[0].plot(t, samples.imag, label="Q")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    # Simple PSD/periodogram
    nfft = 4096
    if samples.size < nfft:
        nfft = 1 << (samples.size - 1).bit_length()
    win = np.hanning(nfft)
    seg = samples[:nfft]
    spec = np.fft.fftshift(np.fft.fft(seg * win))
    psd = 20 * np.log10(np.abs(spec) + 1e-12)
    if samp_rate:
        freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / samp_rate))
        ax[1].plot(freqs, psd)
        ax[1].set_xlabel("Frequency (Hz)")
    else:
        ax[1].plot(psd)
        ax[1].set_xlabel("Bin")
    ax[1].set_ylabel("PSD (dB)")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser(description="Read PlutoSDR .iq files and inspect/plot samples")
    ap.add_argument("--path", required=True, help="Path to .iq file (binary)")
    ap.add_argument("--samp-rate", type=float, default=None, help="Sample rate (for timing/PSD)")
    ap.add_argument("--plot", action="store_true", help="Plot time-domain and PSD")
    ap.add_argument("--plot-n", type=int, default=0, help="Number of samples to plot (0 = all)")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1
    
    samples = read_iq(args.path)

    summarize(samples, args.samp_rate)

    if args.plot:
        plot_wave(samples, args.samp_rate, args.plot_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
