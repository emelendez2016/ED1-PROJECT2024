import os
import pandas as pd
import time


def find_parquet_file(folder, file_id):
    """
    Search for a Parquet file in a given folder by matching the file_id.
    """
    file_path = os.path.join(folder, f"{file_id}.parquet")
    return file_path if os.path.exists(file_path) else None


def extract_event_window(df, offset, samples_per_second, event_duration=50):
    """
    Extracts an event window from the dataframe starting from `offset`, keeping only the
    required number of rows based on `samples_per_second` and `event_duration`.
    
    - offset: Starting row index.
    - samples_per_second: Number of samples per second.
    - event_duration: Duration of the event in seconds (default = 1 sec).
    """
    num_rows = samples_per_second * event_duration  # Define event window length
    return df.iloc[offset : offset + num_rows]  # Extract event window


def label_data(metadata_csv, eeg_folder, spec_folder, processed_eeg_folder, processed_spec_folder, samples_per_second):
    """
    Reads the metadata CSV file, extracts labeled event windows from EEG and spectrogram data,
    and saves them as Parquet files in processed folders.
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_csv)

    for index, row in metadata_df.iterrows():
        # Extract metadata values
        eeg_id, eeg_sub_id, eeg_offset = row["eeg_id"], row["eeg_sub_id"], int(row["eeg_label_offset_seconds"])
        spectrogram_id, spectrogram_sub_id, spectrogram_offset = row["spectrogram_id"], row["spectrogram_sub_id"], int(row["spectrogram_label_offset_seconds"])
        label = row["expert_consensus"]


        eeg_file_path = find_parquet_file(eeg_folder, eeg_id)
        if eeg_file_path:
            eeg_df = pd.read_parquet(eeg_file_path)
            event_eeg = extract_event_window(eeg_df, eeg_offset, samples_per_second)
            eeg_output_path = os.path.join(processed_eeg_folder, f"{eeg_id}_{eeg_sub_id}_{eeg_offset}_{label}.parquet")
            event_eeg.to_parquet(eeg_output_path, index=False)


        spec_file_path = find_parquet_file(spec_folder, spectrogram_id)
        if spec_file_path:
            spec_df = pd.read_parquet(spec_file_path)
            event_spec = extract_event_window(spec_df, spectrogram_offset, samples_per_second)
            spec_output_path = os.path.join(processed_spec_folder, f"{spectrogram_id}_{spectrogram_sub_id}_{spectrogram_offset}_{label}.parquet")
            event_spec.to_parquet(spec_output_path, index=False)


if __name__ == "__main__":

    start = time.time()

    preprocess_data(
        "G:/My Drive/fau/egn4952c_spring_2025/ED1-PROJECT2024/hms-harmful-brain-activity-classification/train.csv",
        "G:/My Drive/fau/egn4952c_spring_2025/ED1-PROJECT2024/hms-harmful-brain-activity-classification/train_eegs/",
        "G:/My Drive/fau/egn4952c_spring_2025/ED1-PROJECT2024/hms-harmful-brain-activity-classification/train_spectrograms/",
        "G:/My Drive/fau/egn4952c_spring_2025/ED1-PROJECT2024/labeled_training_data/labeled_training_eegs/",
        "G:/My Drive/fau/egn4952c_spring_2025/ED1-PROJECT2024/labeled_training_data/labeled_training_spectrograms/",
        200
    )

    end = time.time()
    runtime = f"{((end - start)/3600):02.0f}:{((end - start)/60)%60:02.0f}:{(end - start)%60:02.0f}"
    print(f"runtime: {runtime}")
