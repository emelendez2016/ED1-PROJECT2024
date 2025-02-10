import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EEGDataset(Dataset):
    def __init__(self, processed_eeg_folder):
        self.eeg_files = [os.path.join(processed_eeg_folder, f) for f in os.listdir(processed_eeg_folder) if f.endswith(".parquet")]
        self.labels = [self.extract_label(f) for f in self.eeg_files]  # Extract labels

        # Define label mapping for 6 classes
        self.label_to_index = {
            "Seizure": 0, "LRDA": 1, "GRDA": 2, "LPD": 3, "GPD": 4, "Other": 5 
        }

    def extract_label(self, filename):
        base_name = os.path.basename(filename)
        label = base_name.rsplit("_", 1)[-1].split(".")[0]
        return label

    def __len__(self):
        return len(self.eeg_files)

    def __getitem__(self, idx):
        file_path = self.eeg_files[idx]
        df = pd.read_parquet(file_path).values
        data_sample = torch.tensor(df, dtype=torch.float32)

        label_str = self.labels[idx]  # Get string label
        label = torch.tensor(self.label_to_index[label_str], dtype=torch.long)  # Convert to class index (0-5)

        return data_sample, label


class SpectrogramDataset(Dataset):
    def __init__(self, processed_spec_folder):
        self.spec_files = [os.path.join(processed_spec_folder, f) for f in os.listdir(processed_spec_folder) if f.endswith(".parquet")]
        self.labels = [self.extract_label(f) for f in self.spec_files]  # Extract labels

        # Define label mapping for 6 classes
        self.label_to_index = {
            "Seizure": 0, "LRDA": 1, "GRDA": 2, "LPD": 3, "GPD": 4, "Other": 5 
        }

    def extract_label(self, filename):
        base_name = os.path.basename(filename)
        label = base_name.rsplit("_", 1)[-1].split(".")[0]
        return label

    def __len__(self):
        return len(self.spec_files)

    def __getitem__(self, idx):
        file_path = self.spec_files[idx]
        df = pd.read_parquet(file_path).values
        data_sample = torch.tensor(df, dtype=torch.float32)

        label_str = self.labels[idx]  # Get string label
        label = torch.tensor(self.label_to_index[label_str], dtype=torch.long)  # Convert to class index (0-5)

        return data_sample, label


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes=6):  # Fixed at 6 classes
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)  # Adjust based on spectrogram shape
        self.fc2 = nn.Linear(128, num_classes)  # Ensure 6 output classes

    def forward(self, x):
        x = x.view(x.size(0), 1, 20, 20)  # Adjust shape if needed
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EEGTransformer(nn.Module):
    def __init__(self, input_dim=20, model_dim=128, num_heads=4, num_layers=2, num_classes=6):  # Fixed at 6 classes
        super(EEGTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)  # Ensure 6 output classes

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


def train_transformer(processed_eeg_folder, batch_size, epochs, lr, num_workers=4):
    """Train and save the Transformer model using EEG data."""
    print("\nLoading EEG Data...")
    eeg_dataset = EEGDataset(processed_eeg_folder)
    eeg_loader = DataLoader(eeg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    print("\nInitializing Transformer Model...")
    transformer_model = EEGTransformer(num_classes=6)

    print("\nDefining Loss and Optimizer...")
    criterion = nn.CrossEntropyLoss()
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=lr)

    print("\nTraining Transformer on Raw EEG Data...")
    train_model(transformer_model, eeg_loader, criterion, transformer_optimizer, epochs, model_name="transformer_model")


def train_cnn(processed_spec_folder, batch_size, epochs, lr, num_workers=4):
    """Train and save the CNN model using Spectrogram data."""
    print("\nLoading Spectrogram Data...")
    spec_dataset = SpectrogramDataset(processed_spec_folder)
    spec_loader = DataLoader(spec_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    print("\nInitializing CNN Model...")
    cnn_model = SpectrogramCNN(num_classes=6)

    print("\nDefining Loss and Optimizer...")
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=lr)

    print("\nTraining CNN on Spectrogram Data...")
    train_model(cnn_model, spec_loader, criterion, cnn_optimizer, epochs, model_name="cnn_model")


def train_model(model, train_loader, criterion, optimizer, epochs=5, model_name="model"):
    model.train()

    # Outer progress bar for epochs
    epoch_progress = tqdm(range(epochs), desc="Overall Training Progress", position=0, leave=True)

    for epoch in epoch_progress:
        total_loss = 0
        correct = 0
        total = 0

        # Inner progress bar for batches within each epoch
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", position=1, leave=True)

        for data, labels in batch_progress:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update batch progress bar with live metrics
            batch_progress.set_postfix(loss=total_loss / (total or 1), acc=100 * correct / (total or 1))

        # Update outer progress bar description to show accuracy per epoch
        epoch_progress.set_postfix(epoch_loss=total_loss / len(train_loader), epoch_acc=100 * correct / total)

        print(f"\nEpoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%")

    save_path = f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved as {save_path}")


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct/total:.2f}%")


def main(processed_eeg_folder, processed_spec_folder, batch_size=64, epochs=5, lr=0.001, num_workers=4, run_transformer=True, run_cnn=True):
    """Main function to train models with multiprocessing support."""
    processes = []

    if run_transformer:
        p1 = mp.Process(target=train_transformer, args=(processed_eeg_folder, batch_size, epochs, lr, num_workers))
        p1.start()
        processes.append(p1)

    if run_cnn:
        p2 = mp.Process(target=train_cnn, args=(processed_spec_folder, batch_size, epochs, lr, num_workers))
        p2.start()
        processes.append(p2)

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    main(
        "C:/users/mpm/labeled_training_data/labeled_training_eegs/training_subset/",
        "C:/users/mpm/labeled_training_data/labeled_training_spectrograms/training_subset/",
        batch_size=1,
        epochs=1,
        lr=0.001,
        num_workers=6,
        run_transformer=True,
        run_cnn=False
    )