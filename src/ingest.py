import mne
from mne.datasets import eegbci

def load_eeg_data(subjects=1, runs=1):

    raw_fnames = mne.datasets.eegbci.load_data(subjects=subjects, runs=runs, path='data/')
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True)
    return raw

if __name__ == "__main__":
    print("Loading EEG data")
    raw = load_eeg_data(subjects=1, runs=1)

    print("\n=== EEG DATA INFO ===")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"Channel names: {raw.ch_names[:10]}...")

    data, times = raw.get_data(return_times=True)
    print(f"\nData shape: {data.shape}")
    print(f"(channels × time points)")