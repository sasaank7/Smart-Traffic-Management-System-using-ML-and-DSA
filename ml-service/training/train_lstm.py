"""
LSTM Model Training Script with Real Traffic Data

Trains LSTM model on METR-LA dataset for traffic prediction
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_service.models.lstm_model import LSTMTrafficPredictor
from scripts.data_collection.download_datasets import TrafficDatasetManager


class LSTMTrainer:
    """LSTM model trainer for traffic prediction"""

    def __init__(
        self,
        input_size: int = 207,  # Number of sensors
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'auto',
    ):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = LSTMTrafficPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=input_size,  # Predict all sensors
            dropout=dropout,
        ).to(self.device)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
        }

    def train(
        self,
        train_data: dict,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        patience: int = 10,
        save_dir: str = './models/checkpoints',
    ):
        """
        Train LSTM model

        Args:
            train_data: Dictionary with X_train, y_train, X_val, y_val
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data['X_train']),
            torch.FloatTensor(train_data['y_train'])
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(train_data['X_val']),
            torch.FloatTensor(train_data['y_val'])
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training LSTM on {len(train_loader.dataset)} samples")
        print(f"Validation on {len(val_loader.dataset)} samples")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            val_loss = self.evaluate(val_loader, criterion)

            # Scheduler step
            scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch + 1)

            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)

                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save(self.model.state_dict(), checkpoint_path)

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")

    def evaluate(self, data_loader: DataLoader, criterion) -> float:
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def test(self, test_data: dict, batch_size: int = 64):
        """Test model and compute metrics"""
        test_dataset = TensorDataset(
            torch.FloatTensor(test_data['X_test']),
            torch.FloatTensor(test_data['y_test'])
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()

        self.model.eval()
        total_mse = 0
        total_mae = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)

                mse = criterion(outputs, y_batch)
                mae = mae_criterion(outputs, y_batch)

                total_mse += mse.item()
                total_mae += mae.item()

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_mse = total_mse / len(test_loader)
        avg_mae = total_mae / len(test_loader)
        rmse = np.sqrt(avg_mse)

        # Denormalize errors
        std = test_data.get('std', 1.0)
        rmse_actual = rmse * std
        mae_actual = avg_mae * std

        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  RMSE: {rmse:.4f} (normalized), {rmse_actual:.2f} km/h (actual)")
        print(f"  MAE:  {avg_mae:.4f} (normalized), {mae_actual:.2f} km/h (actual)")
        print(f"{'='*60}\n")

        return {
            'rmse': rmse,
            'mae': avg_mae,
            'rmse_actual': rmse_actual,
            'mae_actual': mae_actual,
            'predictions': np.concatenate(all_predictions),
            'targets': np.concatenate(all_targets),
        }

    def plot_training_history(self, save_path: str = './models/training_history.png'):
        """Plot training history"""
        plt.figure(figsize=(10, 6))

        plt.plot(self.history['epochs'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epochs'], self.history['val_loss'], label='Val Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")

    def save_final_model(self, save_path: str = './models/saved_models/lstm_traffic_v1.pt'):
        """Save final model"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path)
        print(f"Saved final model to {save_path}")

        # Save metadata
        metadata = {
            'model_type': 'LSTM',
            'input_size': self.model.lstm.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'training_date': datetime.now().isoformat(),
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else None,
        }

        metadata_path = save_path.parent / 'lstm_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    """Main training script"""
    print("\n" + "="*60)
    print("LSTM Traffic Prediction Model Training")
    print("="*60 + "\n")

    # Initialize dataset manager
    manager = TrafficDatasetManager(data_dir="./data")

    # Check if processed data exists
    processed_path = manager.processed_dir / "metr_la_processed.npz"

    if not processed_path.exists():
        print("Processed data not found. Downloading and preprocessing METR-LA dataset...")

        # Download dataset
        manager.download_metr_la()

        # Load and preprocess
        data, metadata = manager.load_metr_la()
        processed_data = manager.preprocess_for_lstm(
            data,
            sequence_length=12,  # 1 hour (12 * 5 min)
            horizon=1,  # 5 minutes ahead
        )
        manager.save_processed_data(processed_data)

    else:
        print(f"Loading preprocessed data from {processed_path}...")
        processed_data = manager.load_processed_data()

    print(f"\nDataset Info:")
    print(f"  Train samples: {len(processed_data['X_train'])}")
    print(f"  Val samples:   {len(processed_data['X_val'])}")
    print(f"  Test samples:  {len(processed_data['X_test'])}")
    print(f"  Input shape:   {processed_data['X_train'].shape}")

    # Initialize trainer
    num_sensors = processed_data['X_train'].shape[2]

    trainer = LSTMTrainer(
        input_size=num_sensors,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    )

    # Train model
    trainer.train(
        processed_data,
        epochs=100,
        batch_size=64,
        learning_rate=0.001,
        patience=10,
    )

    # Test model
    test_results = trainer.test(processed_data)

    # Plot training history
    trainer.plot_training_history()

    # Save final model
    trainer.save_final_model()

    print("\n✅ Training completed successfully!")
    print(f"Final Test RMSE: {test_results['rmse_actual']:.2f} km/h")
    print(f"Final Test MAE:  {test_results['mae_actual']:.2f} km/h")


if __name__ == "__main__":
    main()
