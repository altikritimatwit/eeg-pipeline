"""Shared utility functions for EEG pipeline"""

import mne
from mne.datasets import eegbci
import numpy as np

def load_raw_eeg(subjects=1, runs=1, data_path='data/'):
    """Load raw EEG data from PhysioNet dataset"""

    raw_fnames =eegbci.load_data(subjects=subjects, runs=[runs], path='data/')
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
    return raw

def preprocess_eeg(raw, l_freq=1.0, h_freq=40.0, notch_freq=60):
    """Apply preprocessing filter to raw EEG data"""

    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    return raw

def load_and_preprocess(subjects=1, runs=1, data_path='data/'):
    """Load and Preprocess in one function"""

    raw = load_raw_eeg(subjects, runs, data_path)
    raw = preprocess_eeg(raw)
    return raw

def extract_frequency_bands(raw):
    """Extract power in each frequency band"""
    bands = {
        'Delta': (0.5,4), # Deep Sleep
        'Theta': (4,8),     # Drowsiness, meditation
        'Alpha': (8,13),   # Relaxation, eyes closed
        'Beta': (13,30)   # Active thinking, focused
    }

    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        raw_band = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, verbose=False)

        data = raw_band.get_data()
        power = np.mean(data ** 2)
        band_powers[band_name] = power

    return band_powers

def get_recording_metadata(raw):
    """Extract metadata from EEG recording"""

    return{
        'duration': raw.times[-1],
        'n_channels': len(raw.ch_names),
        'sampling_rate': raw.info['sfreq'],
        'channel_names':raw.ch_names
    }