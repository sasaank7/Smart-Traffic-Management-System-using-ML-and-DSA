"""
LSTM Model for Traffic Flow Prediction

Predicts future traffic density based on historical time-series data
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class TrafficDataset(Dataset):
    """PyTorch Dataset for traffic time series"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMTrafficPredictor(nn.Module):
    """
    LSTM Neural Network for Traffic Prediction

    Architecture:
        Input -> LSTM Layers -> Dropout -> Fully Connected -> Output

    Args:
        input_size: Number of input features
        hidden_size: Number of LSTM hidden units
        num_layers: Number of LSTM layers
        output_size: Number of output predictions
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super(LSTMTrafficPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take output from last time step
        out = out[:, -1, :]

        # Dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)

        return out


class TrafficPredictor:
    """
    High-level interface for traffic prediction

    Handles data preprocessing, model inference, and post-processing
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        config: dict = None,
    ):
        self.config = config or self._default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = LSTMTrafficPredictor(
            input_size=self.config["input_features"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)

        # Initialize scaler
        self.scaler = MinMaxScaler()
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)

        self.model.eval()

    def _default_config(self) -> dict:
        """Default model configuration"""
        return {
            "input_features": 7,
            "sequence_length": 12,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
        }

    def preprocess(self, data: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data

        Args:
            data: Raw input array of shape (sequence_length, num_features)

        Returns:
            Preprocessed tensor
        """
        # Normalize
        data_scaled = self.scaler.transform(data)

        # Convert to tensor
        data_tensor = torch.FloatTensor(data_scaled).unsqueeze(0).to(self.device)

        return data_tensor

    def predict(self, sequence: np.ndarray) -> float:
        """
        Predict traffic density for next time step

        Args:
            sequence: Input sequence of shape (sequence_length, num_features)

        Returns:
            Predicted traffic density
        """
        # Preprocess
        x = self.preprocess(sequence)

        # Inference
        with torch.no_grad():
            prediction = self.model(x)

        # Post-process
        pred_value = prediction.cpu().numpy()[0, 0]

        # Denormalize if needed
        return float(pred_value)

    def predict_batch(self, sequences: List[np.ndarray]) -> List[float]:
        """
        Batch prediction for multiple sequences

        Args:
            sequences: List of input sequences

        Returns:
            List of predictions
        """
        # Preprocess all sequences
        x_batch = torch.cat([self.preprocess(seq) for seq in sequences], dim=0)

        # Batch inference
        with torch.no_grad():
            predictions = self.model(x_batch)

        # Post-process
        pred_values = predictions.cpu().numpy().flatten().tolist()

        return pred_values

    def predict_future(
        self,
        initial_sequence: np.ndarray,
        steps: int = 6,
    ) -> List[float]:
        """
        Multi-step ahead prediction

        Args:
            initial_sequence: Initial sequence to start prediction
            steps: Number of future steps to predict

        Returns:
            List of predictions
        """
        predictions = []
        current_sequence = initial_sequence.copy()

        for _ in range(steps):
            # Predict next step
            pred = self.predict(current_sequence)
            predictions.append(pred)

            # Update sequence for next prediction (sliding window)
            # This is a simplified version - in production, you'd update all features
            new_row = current_sequence[-1].copy()
            new_row[0] = pred  # Assuming first feature is traffic density

            current_sequence = np.vstack([current_sequence[1:], new_row])

        return predictions

    def train(
        self,
        train_sequences: np.ndarray,
        train_targets: np.ndarray,
        val_sequences: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        """
        Train the LSTM model

        Args:
            train_sequences: Training sequences (N, sequence_length, features)
            train_targets: Training targets (N, 1)
            val_sequences: Validation sequences
            val_targets: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Create datasets
        train_dataset = TrafficDataset(train_sequences, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_sequences is not None and val_targets is not None:
            val_dataset = TrafficDataset(val_sequences, val_targets)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            # Training
            train_loss = 0
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader, criterion)

                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    def evaluate(self, data_loader: DataLoader, criterion) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                total_loss += loss.item()

        self.model.train()
        return total_loss / len(data_loader)

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")

    def save_scaler(self, path: str):
        """Save data scaler"""
        joblib.dump(self.scaler, path)

    def load_scaler(self, path: str):
        """Load data scaler"""
        self.scaler = joblib.load(path)
