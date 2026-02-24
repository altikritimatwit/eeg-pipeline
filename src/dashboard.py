import mne
from mne.datasets import eegbci
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def load_and_preprocess(subjects=1, runs=1):
    """Load and clean EEG data"""
    raw_fnames = eegbci.load_data(subjects=subjects, runs=[runs], path='data/')
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    raw.filter(l_freq=1.0, h_freq= 40.0)
    raw.notch_filter(freqs=60)
    return raw

def extract_frequence_bands(raw):
    """Extract frequency bands"""
    bands = {
        'Delta': (0.5,4), # Deep Sleep
        'Theta': (4,8),     # Drowsiness, meditation
        'Alpha': (8,13),   # Relaxation, eyes closed
        'Beta': (13,30)   # Active thinking, focused
    }

    band_data = {}

    for band_name, (low_freq, high_freq) in bands.items():
        raw_band = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
        data = raw_band.get_data()
        power = np.mean(data ** 2)
        band_data[band_name] = power

    return band_data


def create_dashboard(raw, band_powers):
    """Create comprehensive EEG analysis dashboard"""

    # Create figure with clean 2x3 grid
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.94, bottom=0.06, left=0.08, right=0.95)

    # 1. Raw signal (top left)
    ax1 = axes[0, 0]
    data, times = raw.get_data(return_times=True, picks=[0])
    ax1.plot(times[:1600], data[0, :1600], color='black', linewidth=0.5)
    ax1.set_title('Raw EEG Signal (Channel 1, First 10s)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (µV)')
    ax1.grid(alpha=0.3)

    # 2. Power Spectral Density (top right)
    ax2 = axes[0, 1]
    psd, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), sfreq=raw.info['sfreq'], fmin=1, fmax=40
    )
    ax2.semilogy(freqs, np.mean(psd, axis=0), color='darkblue', linewidth=2)
    ax2.set_title('Power Spectral Density', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.grid(alpha=0.3)

    # 3. Frequency band powers (middle left)
    ax3 = axes[1, 0]
    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ['purple', 'blue', 'green', 'red']
    ax3.bar(bands, powers, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax3.set_title('Brain Wave Power by Frequency Band', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Average Power (µV²)')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Channel heatmap (middle right)
    ax4 = axes[1, 1]
    data_snippet = raw.get_data()[:, :1000]
    im = ax4.imshow(data_snippet[:20], aspect='auto', cmap='RdBu_r',
                    extent=[0, 1000 / raw.info['sfreq'], 20, 0])
    ax4.set_title('Multi-Channel Activity (First 20 Channels)', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Channel #')
    plt.colorbar(im, ax=ax4, label='Amplitude (µV)', fraction=0.046, pad=0.04)

    # 5. Summary statistics (bottom left)
    ax5 = axes[2, 0]
    ax5.axis('off')

    duration = raw.times[-1]
    n_channels = len(raw.ch_names)
    sfreq = raw.info['sfreq']
    dominant_band = max(band_powers, key=band_powers.get)

    summary_left = f"""Recording Info:
  Duration: {duration:.1f} sec
  Sampling: {sfreq} Hz
  Channels: {n_channels}

Dominant Band: {dominant_band}
    """

    ax5.text(0.1, 0.5, summary_left, fontsize=11, verticalalignment='center',
             family='monospace', transform=ax5.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

    # 6. Band powers detail (bottom right)
    ax6 = axes[2, 1]
    ax6.axis('off')

    summary_right = f"""Frequency Bands:
  Delta: {band_powers['Delta']:.2e} µV²
         Deep sleep

  Theta: {band_powers['Theta']:.2e} µV²
         Meditation

  Alpha: {band_powers['Alpha']:.2e} µV²
         Relaxation

  Beta:  {band_powers['Beta']:.2e} µV²
         Focus
    """

    ax6.text(0.1, 0.5, summary_right, fontsize=11, verticalalignment='center',
             family='monospace', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, pad=1))

    plt.suptitle('EEG Brain Activity Analysis Dashboard', fontsize=16, fontweight='bold')

    plt.savefig('output/eeg_dashboard.png', dpi=150, bbox_inches='tight')
    print("\nDashboard saved to: output/eeg_dashboard.png")
    plt.show()

if __name__ == "__main__":

    import os
    os.makedirs('output', exist_ok=True)

    print("Loading and preprocessing EEG data...")
    raw = load_and_preprocess(subjects=1, runs=1)

    print("Extracting frequency bands...")
    band_powers = extract_frequence_bands(raw)

    print("Creating dashboard...")
    create_dashboard(raw, band_powers)

    print("\nDashboard complete!")