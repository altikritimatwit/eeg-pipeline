import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt

def load_eeg_data(subjects=1, runs=1):
    """Load EEG data"""
    raw_fnames = mne.datasets.eegbci.load_data(subjects=subjects, runs=runs, path='data/')
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    return raw

def plot_raw_signals(raw, duration=10, n_channels=10):
    """Plot raw EEG signals"""

    #first 10 secs of first 10 channels
    raw.plot(
        duration=duration,
        n_channels=n_channels,
        scalings='auto',
        title='Raw EEG Signals'
    )
    plt.show()

def plot_power_spectral_density(raw):
    """Plot frequency content of signals"""

    #which frequency is present in data
    raw.plot_psd(
        fmax=50,
        average=True
    )
    plt.show()

if __name__ == "__main__":
    print("Loading EEG data...")
    raw = load_eeg_data(subjects=1, runs=1)

    print("\nPlotting raw signals...")
    print("Close the plot window to continue...")
    plot_raw_signals(raw, duration=10, n_channels=10)

    print("\nPlotting frequency content (Power Spectral Density)..")
    print("This shows which brain wave frequencies are present...")
    plot_power_spectral_density(raw)