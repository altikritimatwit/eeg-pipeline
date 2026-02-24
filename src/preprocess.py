import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt

def load_eeg_data(subjects=1, runs=1):
    """Load EEG data"""
    raw_fnames = mne.datasets.eegbci.load_data(subjects=subjects, runs=runs, path='data/')
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    return raw

def preprocess_eeg(raw):
    """Clean and filter raw EEG signals"""

    print("Preprocessing EEG data...")

    # remove DC offset and slow drifts
    print("  - Applying high-pass filter(1 Hz)...")
    raw.filter(l_freq=1.0, h_freq=None)

    # remove high-frequency noise
    print("  - Applying low-pass filter(40 Hz)...")
    raw.filter(l_freq=None, h_freq=40.0)

    # remove 60 Hz powerline noise
    print("  - Removing powerline interference (60 Hz)...")
    raw.notch_filter(freqs=60)

    print("Preprocessing complete!\n")
    return raw

if __name__ == "__main__":

    print("Loading raw EEG data...")
    raw = load_eeg_data(subjects=1, runs=1)

    print("\nPlotting before preprocessing...")
    fig1 = raw.plot_psd(fmax=70, average=True, show=False)
    fig1.suptitle('BEFORE preprocessing')

    raw_clean = preprocess_eeg(raw.copy())

    print("\nPlotting after preprocessing...")
    fig2 = raw_clean.plot_psd(fmax=70, average=True, show=False)
    fig2.suptitle('AFTER preprocessing')

    plt.show()

    print("\nCompare the two plots!")
    print("The cleaned signal should look smoother with less noise.")