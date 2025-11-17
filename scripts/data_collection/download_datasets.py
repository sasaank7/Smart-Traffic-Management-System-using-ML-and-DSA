"""
Dataset Download and Management Script

Downloads and prepares real traffic datasets for training:
- METR-LA (Los Angeles): 207 sensors, 34,272 timesteps
- PEMS-BAY (San Francisco): Traffic flow data
- PeMS (California): Comprehensive traffic data
"""
import os
import requests
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import kaggle
from zipfile import ZipFile
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficDatasetManager:
    """Manages traffic dataset downloads and preprocessing"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_metr_la(self) -> Path:
        """
        Download METR-LA dataset from Kaggle

        Dataset: 207 sensors, 34,272 timesteps (4 months), 5-min intervals
        Format: HDF5 file with shape [34272, 207]

        Returns:
            Path to downloaded dataset
        """
        logger.info("Downloading METR-LA dataset from Kaggle...")

        try:
            # Using Kaggle API
            # Requires: pip install kaggle
            # Setup: Place kaggle.json in ~/.kaggle/
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            dataset_name = "annnnguyen/metr-la-dataset"
            download_path = str(self.raw_dir)

            api.dataset_download_files(
                dataset_name, path=download_path, unzip=True
            )

            logger.info("METR-LA dataset downloaded successfully")
            return self.raw_dir / "metr-la.h5"

        except Exception as e:
            logger.error(f"Failed to download from Kaggle: {e}")
            logger.info("Attempting alternative download method...")
            return self._download_metr_la_alternative()

    def _download_metr_la_alternative(self) -> Path:
        """Alternative download method for METR-LA"""
        # Direct download link (from GitHub repo)
        url = "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/metr-la.h5"
        output_path = self.raw_dir / "metr-la.h5"

        logger.info(f"Downloading from {url}...")

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownload progress: {percent:.1f}%", end='')

        print("\nDownload complete!")
        return output_path

    def download_pems_bay(self) -> Path:
        """
        Download PEMS-BAY dataset

        Dataset: San Francisco Bay Area traffic data
        """
        logger.info("Downloading PEMS-BAY dataset...")

        url = "https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/pems-bay.h5"
        output_path = self.raw_dir / "pems-bay.h5"

        response = requests.get(url, stream=True)
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("PEMS-BAY dataset downloaded successfully")
        return output_path

    def load_metr_la(self, file_path: Optional[Path] = None) -> Tuple[np.ndarray, dict]:
        """
        Load METR-LA dataset

        Returns:
            Tuple of (data_array, metadata)
        """
        if file_path is None:
            file_path = self.raw_dir / "metr-la.h5"

        if not file_path.exists():
            logger.warning("Dataset not found. Downloading...")
            file_path = self.download_metr_la()

        logger.info(f"Loading METR-LA from {file_path}...")

        with h5py.File(file_path, 'r') as f:
            # Dataset structure: [timesteps, sensors]
            data = f['speed'][:]  # Shape: (34272, 207)

            metadata = {
                'num_timesteps': data.shape[0],
                'num_sensors': data.shape[1],
                'interval_minutes': 5,
                'total_days': data.shape[0] * 5 / (60 * 24),
                'source': 'METR-LA',
            }

        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Metadata: {metadata}")

        return data, metadata

    def preprocess_for_lstm(
        self,
        data: np.ndarray,
        sequence_length: int = 12,
        horizon: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> dict:
        """
        Preprocess traffic data for LSTM training

        Args:
            data: Raw traffic data (timesteps, sensors)
            sequence_length: Input sequence length (default: 12 = 1 hour)
            horizon: Prediction horizon (default: 1 = 5 minutes ahead)
            train_ratio: Training data ratio
            val_ratio: Validation data ratio

        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Preprocessing data for LSTM...")

        num_samples = data.shape[0] - sequence_length - horizon + 1

        # Create sequences
        X, y = [], []
        for i in range(num_samples):
            # Input sequence: [i:i+sequence_length]
            # Target: [i+sequence_length+horizon-1]
            X.append(data[i:i+sequence_length, :])
            y.append(data[i+sequence_length+horizon-1, :])

        X = np.array(X)  # Shape: (samples, sequence_length, sensors)
        y = np.array(y)  # Shape: (samples, sensors)

        # Split data
        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)

        X_train = X[:n_train]
        y_train = y[:n_train]

        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]

        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]

        # Normalize using training data statistics
        mean = X_train.mean()
        std = X_train.std()

        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std

        y_train = (y_train - mean) / std
        y_val = (y_val - mean) / std
        y_test = (y_test - mean) / std

        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'mean': mean,
            'std': std,
        }

    def save_processed_data(self, data_dict: dict, name: str = "metr_la_processed"):
        """Save preprocessed data"""
        output_path = self.processed_dir / f"{name}.npz"

        np.savez(
            output_path,
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            mean=data_dict['mean'],
            std=data_dict['std'],
        )

        logger.info(f"Saved processed data to {output_path}")
        return output_path

    def load_processed_data(self, name: str = "metr_la_processed") -> dict:
        """Load preprocessed data"""
        file_path = self.processed_dir / f"{name}.npz"

        logger.info(f"Loading processed data from {file_path}...")

        data = np.load(file_path)

        return {
            'X_train': data['X_train'],
            'y_train': data['y_train'],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
            'X_test': data['X_test'],
            'y_test': data['y_test'],
            'mean': float(data['mean']),
            'std': float(data['std']),
        }

    def create_sample_data(self, num_sensors: int = 10, num_days: int = 30) -> np.ndarray:
        """
        Create synthetic traffic data for testing

        Args:
            num_sensors: Number of sensors
            num_days: Number of days to simulate

        Returns:
            Synthetic traffic data
        """
        logger.info(f"Creating synthetic data: {num_sensors} sensors, {num_days} days")

        # 5-minute intervals
        timesteps_per_day = 24 * 60 // 5
        total_timesteps = num_days * timesteps_per_day

        # Generate realistic traffic patterns
        time_of_day = np.arange(total_timesteps) % timesteps_per_day
        day_of_week = (np.arange(total_timesteps) // timesteps_per_day) % 7

        # Base traffic pattern (higher during rush hours)
        base_pattern = 30 + 20 * np.sin(2 * np.pi * time_of_day / timesteps_per_day)

        # Weekend reduction
        weekend_factor = np.where(day_of_week >= 5, 0.7, 1.0)

        # Rush hour peaks
        morning_rush = 15 * np.exp(-((time_of_day - 144)**2) / 200)  # 8 AM
        evening_rush = 20 * np.exp(-((time_of_day - 216)**2) / 200)  # 6 PM

        # Combine patterns
        data = np.zeros((total_timesteps, num_sensors))
        for i in range(num_sensors):
            sensor_base = base_pattern + morning_rush + evening_rush
            sensor_base *= weekend_factor

            # Add sensor-specific variation
            sensor_variance = np.random.normal(0, 5, total_timesteps)
            data[:, i] = sensor_base + sensor_variance

            # Ensure non-negative
            data[:, i] = np.maximum(data[:, i], 0)

        return data


def main():
    """Main execution"""
    manager = TrafficDatasetManager(data_dir="./data")

    print("\n=== Traffic Dataset Manager ===\n")
    print("1. Download METR-LA dataset")
    print("2. Download PEMS-BAY dataset")
    print("3. Preprocess METR-LA for training")
    print("4. Create synthetic data")
    print("5. Download and preprocess all")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == "1":
        manager.download_metr_la()

    elif choice == "2":
        manager.download_pems_bay()

    elif choice == "3":
        data, metadata = manager.load_metr_la()
        processed = manager.preprocess_for_lstm(data)
        manager.save_processed_data(processed)

    elif choice == "4":
        data = manager.create_sample_data(num_sensors=10, num_days=30)
        processed = manager.preprocess_for_lstm(data)
        manager.save_processed_data(processed, name="synthetic_data_processed")

    elif choice == "5":
        # Download METR-LA
        manager.download_metr_la()

        # Load and preprocess
        data, metadata = manager.load_metr_la()
        processed = manager.preprocess_for_lstm(data)
        manager.save_processed_data(processed)

        print("\n✅ All datasets downloaded and preprocessed!")

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()
