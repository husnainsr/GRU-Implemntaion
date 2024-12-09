import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence

import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return timestamp

def save_metrics(metrics, timestamp):
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    metrics_file = f'metrics/metrics_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_file}")

def plot_training_history(history, timestamp):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training plots saved to plots/training_history_{timestamp}.png")

class DVECDataset(Dataset):
    def __init__(self, root_dir, is_real=True):
        self.root_dir = os.path.join(root_dir, 'real' if is_real else 'fake')
        self.samples = []
        self.labels = []
        
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    self.samples.append(file_path)
                    self.labels.append(1 if is_real else 0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        data = np.load(self.samples[idx])
        
        # Check for NaN or Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            logging.warning(f"Found NaN or Inf in file: {self.samples[idx]}")
            # Replace NaN/Inf with 0 or some other value
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add data statistics logging for first few files
        if idx < 5:
            logging.info(f"File: {self.samples[idx]}")
            logging.info(f"Data stats - Min: {data.min():.4f}, Max: {data.max():.4f}, Mean: {data.mean():.4f}, Std: {data.std():.4f}")
        
        data = torch.FloatTensor(data)
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

def collate_fn(batch):
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    data_padded = pad_sequence(data, batch_first=True)
    labels = torch.stack(labels)
    return data_padded, labels

class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dropout_input = nn.Dropout(p=0.4)
        self.gru = nn.GRU(input_size, 64, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(64, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dropout_input(x)
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        out, _ = self.gru(x, h0)
        
        out = out[:, -1, :]
        
        out = self.dropout1(out)
        out = self.fc1(out)
        
        out = self.dropout2(out)
        out = self.fc2(out)
        
        out = self.log_softmax(out)
        return out

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if batch_idx == 0:
            logging.info(f"Input batch stats - Min: {inputs.min():.4f}, Max: {inputs.max():.4f}, Mean: {inputs.mean():.4f}")
            logging.info(f"Labels: {labels}")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if batch_idx == 0:
            logging.info(f"Model output stats - Min: {outputs.min():.4f}, Max: {outputs.max():.4f}, Mean: {outputs.mean():.4f}")
            logging.info(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
        
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            logging.error(f"NaN loss detected at batch {batch_idx}")
            logging.error(f"Outputs: {outputs}")
            logging.error(f"Labels: {labels}")
            continue
        
        loss.backward()
        
        if batch_idx == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        logging.error(f"NaN/Inf gradient detected in {name}")
                    else:
                        logging.info(f"Gradient norm for {name}: {grad_norm:.4f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss, model, timestamp):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info('Early stopping triggered')
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logging.info(f'EarlyStopping counter reset: validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            self.val_loss_min = val_loss

def main():
    timestamp = setup_logging()
    logging.info("Starting training process...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    data_dir = "D:/FYP/Data Aqsa/Implementation/DVECs2"
    logging.info(f"Loading data from: {data_dir}")
    
    logging.info("Loading real and fake datasets...")
    real_dataset = DVECDataset(data_dir, is_real=True)
    fake_dataset = DVECDataset(data_dir, is_real=False)
    logging.info(f"Real dataset size: {len(real_dataset)}")
    logging.info(f"Fake dataset size: {len(fake_dataset)}")
    
    full_dataset = torch.utils.data.ConcatDataset([real_dataset, fake_dataset])
    logging.info(f"Total dataset size: {len(full_dataset)}")
    
    train_data, val_data = train_test_split(full_dataset, test_size=0.2, random_state=42)
    logging.info(f"Training set size: {len(train_data)}")
    logging.info(f"Validation set size: {len(val_data)}")
    
    train_loader = DataLoader(
        train_data, 
        batch_size=32, 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    first_batch = next(iter(train_loader))
    logging.info(f"First batch shapes - Input: {first_batch[0].shape}, Labels: {first_batch[1].shape}")
    
    input_size = first_batch[0].shape[-1]  # Get the feature dimension
    model = GRUModel(input_size=input_size).to(device)
    logging.info(f"Model initialized with input size: {input_size}")
    logging.info(f"Model architecture:\n{model}")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    logging.info("Using NLLLoss and Adam optimizer with lr=1e-2")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    early_stopping = EarlyStopping(patience=7, verbose=True)
    
    num_epochs = 50
    best_val_acc = 0.0
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'models/best_model_{timestamp}.pth')
            logging.info(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        
        early_stopping(val_loss, model, timestamp)
        if early_stopping.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    final_metrics = {
        'training_history': history,
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_acc,
        'final_val_accuracy': val_acc,
        'num_epochs_trained': epoch + 1,
        'early_stopping_triggered': early_stopping.early_stop,
        'timestamp': timestamp
    }
    
    logging.info("Training completed. Saving final metrics and plots...")
    save_metrics(final_metrics, timestamp)
    plot_training_history(history, timestamp)
    
    logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    if early_stopping.early_stop:
        logging.info(f"Training stopped early at epoch {epoch+1}")

if __name__ == "__main__":
    main()