from utils import load_and_preprocess, extract_frequency_bands
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Loading and preprocessing EEG data...")
    raw = load_and_preprocess(subjects=1, runs=1)

    print("\nExtracting frequency bands...")
    band_powers = extract_frequency_bands(raw)

    print("\n=== FREQUENCY BAND POWERS ===")
    for band_name, power in band_powers.items():
        print(f"{band_name}: {power:.2e} µV²")

    # Plot
    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ['purple', 'blue', 'green', 'red']

    plt.figure(figsize=(10,6))
    plt.bar(bands, powers, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Average Power (µV²)')
    plt.title('Brain Wave Power by Frequency Band')
    plt.show()