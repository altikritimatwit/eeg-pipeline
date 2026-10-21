from utils import load_raw_eeg, get_recording_metadata

if __name__ == "__main__":
    print("Loading EEG data")
    raw = load_raw_eeg(subjects=1, runs=1)
    metadata = get_recording_metadata(raw)

    print("\n=== EEG DATA INFO ===")
    print(f"Number of channels: {metadata['n_channels']}")
    print(f"Sampling rate: {metadata['sampling_rate']} Hz")
    print(f"Duration: {metadata['duration']:.2f} seconds")
    print(f"Channel names: {metadata['channel_names'][:10]}...")

    data, times = raw.get_data(return_times=True)
    print(f"\nData shape: {data.shape}")
    print(f"(channels × time points)")