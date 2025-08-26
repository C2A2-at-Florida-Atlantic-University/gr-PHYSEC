#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False


def read_fc32(path: str, count: int | None, skip: int) -> np.ndarray:
    # fc32 (complex64): interleaved float32 I/Q compatible with numpy complex64
    bytes_per_sample = 8  # 2 * float32
    offset = skip * bytes_per_sample
    data = np.fromfile(path, dtype=np.complex64, count=count, offset=offset)
    return data


def read_sc16(path: str, count: int | None, skip: int, iq_order: str = "iq") -> np.ndarray:
    # sc16: interleaved int16 I/Q → complex64 normalized to [-1, 1)
    # count refers to complex samples
    if count is None:
        raw = np.fromfile(path, dtype=np.int16)
    else:
        # each complex sample = 2 int16
        raw = np.fromfile(path, dtype=np.int16, count=count * 2, offset=skip * 4)
    if raw.size % 2 != 0:
        raw = raw[: raw.size - (raw.size % 2)]
    iq = raw.reshape(-1, 2)
    if iq_order.lower() == "iq":
        i = iq[:, 0].astype(np.float32)
        q = iq[:, 1].astype(np.float32)
    else:
        q = iq[:, 0].astype(np.float32)
        i = iq[:, 1].astype(np.float32)
    scale = 32768.0
    c = (i / scale) + 1j * (q / scale)
    return c


def summarize(samples: np.ndarray, samp_rate: float | None) -> None:
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


def plot_wave(samples: np.ndarray, samp_rate: float | None, num: int) -> None:
    if not HAVE_PLOT:
        print("matplotlib not available; skipping plots.")
        return
    num = min(num, samples.size)
    t = np.arange(num) / (samp_rate if samp_rate else 1.0)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, samples[:num].real, label="I")
    ax[0].plot(t, samples[:num].imag, label="Q", alpha=0.7)
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
    ap.add_argument("--format", choices=["fc32", "sc16"], default="fc32", help="Sample storage format")
    ap.add_argument("--iq-order", choices=["iq", "qi"], default="iq", help="Order in interleaved pairs (for sc16)")
    ap.add_argument("--skip", type=int, default=0, help="Samples to skip (complex samples)")
    ap.add_argument("--count", type=int, default=None, help="Number of complex samples to read")
    ap.add_argument("--samp-rate", type=float, default=None, help="Sample rate (for timing/PSD)")
    ap.add_argument("--plot", action="store_true", help="Plot time-domain and PSD")
    ap.add_argument("--plot-n", type=int, default=4096, help="Number of samples to plot")
    ap.add_argument("--save-npy", default=None, help="Optional path to save numpy .npy of complex64 samples")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found: {args.path}", file=sys.stderr)
        return 1

    if args.format == "fc32":
        samples = read_fc32(args.path, args.count, args.skip)
    else:
        samples = read_sc16(args.path, args.count, args.skip, args.iq_order)

    summarize(samples, args.samp_rate)

    if args.save_npy:
        np.save(args.save_npy, samples.astype(np.complex64))
        print(f"Saved: {args.save_npy}")

    if args.plot:
        plot_wave(samples, args.samp_rate, args.plot_n)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
