from utils import load_raw_eeg, preprocess_eeg
import matplotlib.pyplot as plt

if __name__ == "__main__":

    print("Loading raw EEG data...")
    raw = load_raw_eeg(subjects=1, runs=1)

    print("\nPlotting before preprocessing...")
    fig1 = raw.plot_psd(fmax=70, average=True, show=False)
    fig1.suptitle('BEFORE preprocessing')

    #Processing
    raw_clean = preprocess_eeg(raw.copy())

    print("\nPlotting after preprocessing...")
    fig2 = raw_clean.plot_psd(fmax=70, average=True, show=False)
    fig2.suptitle('AFTER preprocessing')

    plt.show()
