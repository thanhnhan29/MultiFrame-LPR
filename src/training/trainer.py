"""Trainer class encapsulating the training and validation loop."""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.postprocess import decode_with_confidence


class Trainer:
    """Encapsulates training, validation, and inference logic."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        """
        Args:
            model: The neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader (can be None).
            config: Configuration object with training parameters.
            idx2char: Index to character mapping for decoding.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        
        # Loss and optimizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS
        )
        self.scaler = GradScaler()
        
        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        
        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                preds = self.model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)

            # Scale loss & backward
            self.scaler.scale(loss).backward()
            
            # Unscale (required before gradient clipping)
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            # Save scale before step for scheduler check
            scale_before = self.scaler.get_scale()
            
            # Step optimizer & update scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Step scheduler only if optimizer actually stepped (scale not reduced)
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})
        
        return epoch_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float, List[str]]:
        """Run validation and generate submission data."""
        if self.val_loader is None:
            return 0.0, 0.0, []
        
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []
        
        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)
                
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                val_loss += loss.item()
                
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    
                total_samples += len(labels_text)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        return avg_val_loss, val_acc, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file."""
        with open(self.config.SUBMISSION_FILE, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {self.config.SUBMISSION_FILE}")

    def save_model(self, path: str = "best_model.pth") -> None:
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        """Run the full training loop."""
        print(f"🚀 TRAINING START | Device: {self.device}")
        
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            
            # Training
            avg_train_loss = self.train_one_epoch()
            
            # Validation
            avg_val_loss, val_acc, submission_data = self.validate()
            
            print(f"Result: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model("best_model.pth")
                print(f" -> ⭐ Saved Best Model! ({val_acc:.2f}%)")
                
                if submission_data:
                    self.save_submission(submission_data)

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader.
        
        Returns:
            List of (track_id, predicted_text, confidence) tuples.
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        
        with torch.no_grad():
            for images, _, _, _, track_ids in loader:
                images = images.to(self.device)
                preds = self.model(images)
                
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        return results
