"""
train.py - Training Pipeline
==============================
Model training, validation, early stopping, learning rate scheduling,
and saving training history.

Usage:
    python train.py
    veya
    python main.py --mode train
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from data.dataset import (get_dataloaders, get_class_weights,
                          get_dataloaders_for_dataset,
                          get_class_weights_for_dataset)
from models.mini_xception import get_model
from models.efficientnet import get_efficientnet_model
from models.resnet import get_resnet_model
from models.hsemotion_model import get_hsemotion_model
from utils.visualization import plot_training_history


class EarlyStopping:
    """
    Early Stopping: Stops training if validation loss does not improve.

    This is one of the most effective techniques for preventing overfitting.
    The model is saved at the epoch where it achieves
    noktada kaydedilir.

    Args:
        patience (int): Kaç epoch iyileşme olmazsa dur
        min_delta (float): Minimum iyileşme miktarı
    """

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            # İyileşme yok
            self.counter += 1
            print(f"  [Early Stopping] {self.counter}/{self.patience} "
                  f"(en iyi: {self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  [Early Stopping] Training stopped!")
        else:
            # İyileşme var
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """
    Model training and validation manager.

    Features:
    - CrossEntropyLoss (with class weights)
    - Adam optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Save best modelme
    - Eğitim geçmişini JSON olarak kaydetme
    """

    def __init__(self, model=None, dataloaders=None, dataset_name="fer2013", model_name="mini_xception"):
        """
        Initialize the Trainer.

        Args:
            model: PyTorch model (auto-created if None)
            dataloaders: DataLoader dict (auto-created if None)
            dataset_name: Dataset to train on
            model_name: Model architecture (mini_xception, efficientnet, resnet, hsemotion)
        """
        self.dataset_name = dataset_name
        self.model_name = model_name

        # Get model-specific config
        model_cfg = config.MODEL_CONFIGS.get(model_name, config.MODEL_CONFIGS["mini_xception"])

        # Model
        if model is None:
            if model_name == "efficientnet":
                self.model = get_efficientnet_model(
                    freeze_backbone=config.EFFICIENTNET_FREEZE_BACKBONE,
                    unfreeze_last_n=config.EFFICIENTNET_UNFREEZE_LAST_N
                )
            elif model_name == "resnet":
                self.model = get_resnet_model(
                    freeze_backbone=config.RESNET_FREEZE_BACKBONE,
                    unfreeze_last_n=config.RESNET_UNFREEZE_LAST_N
                )
            elif model_name == "hsemotion":
                self.model = get_hsemotion_model()
            else:
                self.model = get_model()
        else:
            self.model = model
        self.model.to(config.DEVICE)

        # DataLoaders (model-specific image size and channels)
        if dataloaders is None:
            self.dataloaders = get_dataloaders_for_dataset(
                dataset_name, model_name=model_name
            )
        else:
            self.dataloaders = dataloaders

        # Loss function (with class weights for imbalanced data)
        class_weights = get_class_weights_for_dataset(dataset_name)
        class_weights = class_weights.to(config.DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer: Adam with model-specific learning rate
        lr = model_cfg["lr"]
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.LR_PATIENCE,
            factor=config.LR_FACTOR
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }

        self.best_val_acc = 0.0

    def train_one_epoch(self, epoch):
        """
        Performs training for a single epoch.

        Returns:
            tuple: (average loss, accuracy percentage)
        """
        self.model.train()  # Eğitim moduna geç (dropout aktif)

        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm ile ilerleme çubuğu
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {epoch}/{config.EPOCHS} [Eğitim]",
            leave=False
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            # Veriyi cihaza (CPU/GPU) taşı
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass ve ağırlık güncelleme
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrikleri güncelle
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # İlerleme çubuğunu güncelle
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.1f}%'
            })

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    @torch.no_grad()  # Doğrulamada gradyan hesaplamaya gerek yok
    def validate(self):
        """
        Evaluates model on the validation set.

        Returns:
            tuple: (average loss, accuracy percentage)
        """
        self.model.eval()  # Değerlendirme moduna geç (dropout kapalı)

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.dataloaders['val'],
            desc="           [Doğrulama]",
            leave=False
        )

        for images, labels in pbar:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, is_best=False):
        """
        Saves a model checkpoint.

        Saved information:
        - model_state_dict: Model weights
        - optimizer_state_dict: Optimizer state (for resuming training)
        - epoch: Epoch at which it was saved
        - best_val_acc: Best validation accuracy
        - config: Training configuration
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': {
                'model_name': self.model_name,
                'batch_size': config.MODEL_CONFIGS.get(
                    self.model_name, {}).get('batch_size', config.BATCH_SIZE),
                'learning_rate': config.MODEL_CONFIGS.get(
                    self.model_name, {}).get('lr', config.LEARNING_RATE),
                'num_classes': config.NUM_CLASSES,
                'img_size': config.MODEL_CONFIGS.get(
                    self.model_name, {}).get('img_size', config.IMG_SIZE)
            }
        }

        # Model-spesifik kayıt yolları
        best_path = config.BEST_MODEL_PATHS.get(self.model_name, config.BEST_MODEL_PATH)
        last_path = config.LAST_MODEL_PATHS.get(self.model_name, config.LAST_MODEL_PATH)

        # Son modeli her zaman kaydet
        torch.save(checkpoint, last_path)

        # En iyi modeli ayrıca kaydet
        if is_best:
            torch.save(checkpoint, best_path)
            print(f"  [BEST] Model saved! (Acc: {self.best_val_acc:.2f}%)")

    def train(self):
        """
        Runs the full training loop.

        Returns:
            dict: Training history (loss, accuracy, lr)
        """
        ds_name = config.DATASET_CONFIGS.get(
            self.dataset_name, {}).get("name", self.dataset_name)
        model_cfg = config.MODEL_CONFIGS.get(self.model_name, config.MODEL_CONFIGS["mini_xception"])
        print("\n" + "=" * 60)
        print(f"  MODEL TRAINING ({ds_name} + {self.model_name})")
        print("=" * 60)
        print(f"  Device:        {config.DEVICE}")
        print(f"  Epochs:        {config.EPOCHS}")
        print(f"  Batch size:    {model_cfg['batch_size']}")
        print(f"  Learning rate: {model_cfg['lr']}")
        print(f"  Image size:    {model_cfg['img_size']}x{model_cfg['img_size']}")
        print(f"  Channels:      {model_cfg['num_channels']}")
        print("=" * 60 + "\n")

        start_time = time.time()

        for epoch in range(1, config.EPOCHS + 1):
            epoch_start = time.time()

            # Eğitim
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Doğrulama
            val_loss, val_acc = self.validate()

            # Geçmişe kaydet
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            # Epoch süresini hesapla
            epoch_time = time.time() - epoch_start

            # Sonuçları yazdır
            print(f"\nEpoch {epoch}/{config.EPOCHS} ({epoch_time:.1f}s)")
            print(f"  Train  -> Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val    -> Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best)

            # Learning Rate Scheduler güncelle
            self.scheduler.step(val_loss)

            # Early Stopping kontrolü
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n[INFO] Eğitim {epoch}. epoch'ta durduruldu (early stopping)")
                break

        # Toplam süre
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        print("\n" + "=" * 60)
        print(f"  TRAINING COMPLETED ({self.model_name})")
        print(f"  Total time: {minutes}m {seconds}s")
        print(f"  Best val accuracy: {self.best_val_acc:.2f}%")
        best_path = config.BEST_MODEL_PATHS.get(self.model_name, config.BEST_MODEL_PATH)
        print(f"  Best model: {best_path}")
        print("=" * 60 + "\n")

        # Eğitim grafiklerini çiz
        plot_training_history(self.history)

        # Save history as JSON (with dataset and model name)
        history_filename = f"training_history_{self.dataset_name}_{self.model_name}.json"
        history_path = os.path.join(config.OUTPUT_DIR, history_filename)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[INFO] Eğitim geçmişi kaydedildi: {history_path}")

        return self.history


def main(dataset_name="fer2013", model_name="mini_xception"):
    """For direct execution."""
    trainer = Trainer(dataset_name=dataset_name, model_name=model_name)
    trainer.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emotion Recognition Training")
    parser.add_argument("--dataset", type=str, default="fer2013",
                        choices=["fer2013", "ferplus", "rafdb", "ckplus"],
                        help="Training dataset (default: fer2013)")
    parser.add_argument("--model", type=str, default="mini_xception",
                        choices=["mini_xception", "efficientnet", "resnet", "hsemotion"],
                        help="Model architecture (default: mini_xception)")
    args = parser.parse_args()
    main(dataset_name=args.dataset, model_name=args.model)
