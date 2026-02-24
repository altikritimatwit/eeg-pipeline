
import mne
from mne.datasets import eegbci
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess(subjects=1, runs=1):
    raw_fnames = eegbci.load_data(subjects=subjects, runs=[runs], path='data/')
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)

    raw.filter(l_freq=1.0, h_freq=40.0)
    raw.notch_filter(freqs=60)

    return raw

def extract_frequency_bands(raw):
    """Extract alpha, beta, theta, delta frequency bands"""

    print("Extracting frequency bands...")

    bands = {
        'Delta (0.5-4 Hz)': (0.5,4), # Deep Sleep
        'Theta (4-8 Hz)': (4,8),     # Drowsiness, meditation
        'Alpha (8-13 Hz)': (8,13),   # Relaxation, eyes closed
        'Beta (13-30 Hz)': (13,30)   # Active thinking, focused
    }

    band_data = {}

    for band_name, (low_freq, high_freq) in bands.items():
        print(f"   Extracting {band_name}...")

        #Filter to this freq. range
        raw_band = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)
        data = raw_band.get_data()

        #Calculate avg power across all channels
        power = np.mean(data ** 2)

        band_data[band_name] = {
            'raw': raw_band,
            'power': power
        }

    return band_data

def plot_frequency_bands(band_data):
    """Plot power in each frequency band"""

    band_names = list(band_data.keys())
    powers = [band_data[band]['power'] for band in band_names]

    plt.figure(figsize=(10,6))
    plt.bar(range(len(band_names)), powers, color=['purple','blue','green','red'])
    plt.xticks(range(len(band_names)), band_names, rotation=45, ha='right')
    plt.ylabel('Average Power (µV²)')
    plt.title('Brain Wave Power by Frequency Band')
    plt.tight_layout()
    plt.show()

def compare_bands_visualization(band_data):
    """Show what each freq. band looks like"""

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    for idx, (band_name, data) in enumerate(band_data.items()):
        raw_band = data['raw']
        signal, times = raw_band.get_data(return_times=True)

        axes[idx].plot(times[:800], signal[0, :800])
        axes[idx].set_title(band_name)
        axes[idx].set_ylabel('Amplitude (µV)')

        if idx == 3:
            axes[idx].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.suptitle('Frequency Band Waveforms (Channel 1, First 5 seconds)', y=1.02)
    plt.show()

if __name__ == "__main__":
    print("Loading and preprocessing EEG data...")
    raw = load_and_preprocess(subjects=1, runs=1)

    print("\nExtracting frequency bands...")
    band_data = extract_frequency_bands(raw)

    print("\n=== FREQUENCY BAND POWERS ===")
    for band_name, data in band_data.items():
        print(f"{band_name}: {data['power']:.2e} µV²")

    print("\nPlotting band powers...")
    plot_frequency_bands(band_data)

    print("\nPlotting band waveforms...")
    compare_bands_visualization(band_data)

    print("\nDone! Check the plots.")