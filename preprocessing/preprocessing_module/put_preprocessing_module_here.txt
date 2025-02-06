# EEG Preprocessing with Parallel Processing + Correct ICA (Multi-Channel): Kevin USE THIS

import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt, iirnotch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import mne
from mne.preprocessing import ICA

# 📌 Define directories
raw_data_dir = r"C:\Users\Kevin Tran\Documents\GitHub ED1\hms-harmful-brain-activity-classificationtrain_eegs\train_eegs"
processed_data_dir = r"C:\Users\Kevin Tran\Documents\Project Data\processed_eegs"
os.makedirs(processed_data_dir, exist_ok=True)  # Ensure processed folder exists

# 📌 Verify if the EEG folder exists
if not os.path.exists(raw_data_dir):
    raise FileNotFoundError(f"🚨 ERROR: The directory {raw_data_dir} does not exist. Please check the path.")

# 📌 List all files to process
all_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith(".parquet")]

# 📌 Check if there are files to process
if len(all_files) == 0:
    raise FileNotFoundError(f"🚨 ERROR: No .parquet files found in {raw_data_dir}. Please check the folder contents.")

print(f"✅ Found {len(all_files)} .parquet files in {raw_data_dir}. Ready to process.\n")

# 📌 Define EEG Preprocessing Functions
def apply_notch_filter(signal, fs=200, freq=60.0, quality_factor=30):
    """Apply a notch filter to remove 60Hz noise."""
    b, a = iirnotch(w0=freq, Q=quality_factor, fs=fs)
    return filtfilt(b, a, signal)

def apply_bandpass_filter(signal, fs=200, lowcut=0.5, highcut=40.0, order=5):
    """Apply a bandpass filter to keep frequencies between 0.5Hz and 40Hz."""
    nyquist = 1.0 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    """Normalize EEG signal to have zero mean and unit variance."""
    return (signal - np.mean(signal)) / np.std(signal)

# 📌 Fixed ICA Function (Applies ICA to Entire Multi-Channel EEG Recording)
def apply_ica(signal_df, fs=200, n_components=10):
    """
    Applies Independent Component Analysis (ICA) to remove artifacts.
    
    Args:
        signal_df: EEG signal data as a Pandas DataFrame (Multi-Channel).
        fs: Sampling frequency (Hz).
        n_components: Number of ICA components to extract.
    
    Returns:
        Cleaned EEG signal as a Pandas DataFrame.
    """
    try:
        # Ensure n_components does not exceed the number of channels
        n_components = min(n_components, len(signal_df.columns))

        # Convert DataFrame to MNE Raw format
        info = mne.create_info(ch_names=list(signal_df.columns), sfreq=fs, ch_types=["eeg"] * len(signal_df.columns))
        raw = mne.io.RawArray(signal_df.values.T, info)  # MNE expects (channels, timepoints)

        # 🔥 Apply a 1.0 Hz high-pass filter before ICA (Fixes the warning)
        raw.filter(l_freq=1.0, h_freq=None, fir_design="firwin")

        # Apply ICA
        ica = ICA(n_components=n_components, random_state=42, max_iter="auto")
        ica.fit(raw)  # Fit ICA on EEG data
        raw_cleaned = ica.apply(raw)  # Apply ICA artifact removal

        # Convert back to DataFrame
        cleaned_signal = raw_cleaned.get_data().T  # Convert (channels, timepoints) → (timepoints, channels)
        return pd.DataFrame(cleaned_signal, columns=signal_df.columns)

    except Exception as e:
        print(f"❌ ICA failed: {e}")
        return signal_df  # Return original signal if ICA fails


# 📌 Function to preprocess a single EEG file
def process_single_file(file_path):
    try:
        # Load EEG Data
        data = pd.read_parquet(file_path)

        # Preprocess each EEG channel
        for channel in data.columns:
            data[channel] = apply_notch_filter(data[channel].values)  # Remove powerline noise
            data[channel] = apply_bandpass_filter(data[channel].values)  # Keep only relevant frequencies
            data[channel] = normalize_signal(data[channel].values)  # Normalize the signal

        # Apply ICA to entire multi-channel EEG recording
        processed_df_cleaned = apply_ica(data, fs=200)

        # Save the processed data
        output_file = os.path.join(processed_data_dir, os.path.basename(file_path))
        processed_df_cleaned.to_parquet(output_file)
        return f"✅ Success: {os.path.basename(file_path)}"
    
    except Exception as e:
        return f"❌ Failed: {os.path.basename(file_path)} | Error: {e}"

# 📌 Threaded Batch Processing
def batch_process_files_threading(file_list, num_workers=os.cpu_count() // 4):
    batch_size = 100  # Process files in batches of 100
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i : i + batch_size]
        print(f"🚀 Processing Batch {i//batch_size+1}/{len(file_list)//batch_size+1} ({len(batch)} files)...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(process_single_file, batch), total=len(batch), desc=f"Batch {i//batch_size+1}"))

        # Summary of results for the current batch
        success = [res for res in results if res.startswith("✅")]
        failed = [res for res in results if res.startswith("❌")]
        print(f"✔️ Batch Summary: {len(success)} Success, {len(failed)} Failed\n")
        if failed:
            print(f"❌ Failed Files: {'; '.join(failed)}\n")

# 📌 Run Preprocessing
if __name__ == "__main__":
    print(f"Starting threaded processing for {len(all_files)} files...\n")
    batch_process_files_threading(all_files)
    print("✅ All EEG files have been processed and saved.\n")

