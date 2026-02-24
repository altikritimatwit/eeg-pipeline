import mne
from mne.datasets import eegbci

print("Downloading EEG data from PhysioNet...")
print("This will take a minute...\n")

# Download subject 1, run 1 (baseline, eyes open)
raw_fnames = eegbci.load_data(subjects=1, runs=[1], path='data/')

print(f"\nDownload complete!")
print(f"Files downloaded: {raw_fnames}")