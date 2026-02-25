"""Distillation Trainer for Teacher-Student Knowledge Distillation training."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class DistillationTrainer:
    """Trainer for Teacher-Student Knowledge Distillation.
    
    Trains a Student model (receiving LR images) to mimic a frozen Teacher model
    (receiving HR images) at two levels:
    1. Feature-level: MSE between fused features after Attention Fusion
    2. Logit-level: KL Divergence between softened output distributions
    
    Combined loss:
        L_total = L_CTC(student, labels) 
                + alpha * L_feature(student_feat, teacher_feat)
                + beta  * L_KD(student_logits, teacher_logits, temperature)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
    ):
        """
        Args:
            teacher: Pre-trained teacher model (will be frozen).
            student: Student model to train.
            train_loader: DataLoader returning (lr_images, hr_images, targets, ...).
            val_loader: Validation DataLoader (optional, uses original dataset format).
            config: Configuration object with training and distillation params.
            idx2char: Index to character mapping for decoding.
        """
        self.teacher = teacher
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)
        
        # Freeze teacher completely
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Losses
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        self.mse_loss = nn.MSELoss()
        
        # Distillation hyperparameters
        self.alpha = getattr(config, 'DISTILL_ALPHA', 1.0)
        self.beta = getattr(config, 'DISTILL_BETA', 0.5)
        self.temperature = getattr(config, 'DISTILL_TEMPERATURE', 4.0)
        
        # Optimizer for Student only
        self.optimizer = optim.AdamW(
            student.parameters(),
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
        self.best_acc = -1.0
        self.current_epoch = 0
    
    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file in configured directory."""
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        """Get experiment name from config."""
        return getattr(self.config, 'EXPERIMENT_NAME', 'distill')
    
    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Knowledge Distillation loss using KL Divergence with temperature.
        
        Args:
            student_logits: Log-softmax output from student [B, T, C]
            teacher_logits: Log-softmax output from teacher [B, T, C]
            
        Returns:
            KD loss scalar.
        """
        T = self.temperature
        
        # Student logits are already log-softmax. Undo, apply temperature, re-softmax
        # student_logits = log_softmax(raw) => raw ≈ student_logits (approximation)
        # For KD we need: student soft = log_softmax(raw/T), teacher soft = softmax(raw/T)
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        
        # KL(teacher || student) averaged over batch and time
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
        return kd_loss
    
    def train_one_epoch(self) -> Dict[str, float]:
        """Train student for one epoch with distillation."""
        self.student.train()
        self.teacher.eval()
        
        epoch_ctc = 0.0
        epoch_feat = 0.0
        epoch_kd = 0.0
        epoch_total = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Distill Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        
        for lr_images, hr_images, targets, target_lengths, _, _ in pbar:
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                # Teacher forward (no grad, HR input)
                with torch.no_grad():
                    teacher_logits, teacher_features = self.teacher(hr_images, return_features=True)
                
                # Student forward (LR input)
                student_logits, student_features = self.student(lr_images, return_features=True)
                
                # 1. CTC Loss (student vs ground truth labels)
                preds_permuted = student_logits.permute(1, 0, 2)  # [T, B, C]
                input_lengths = torch.full(
                    size=(lr_images.size(0),),
                    fill_value=student_logits.size(1),
                    dtype=torch.long
                )
                loss_ctc = self.ctc_loss(preds_permuted, targets, input_lengths, target_lengths)
                
                # 2. Feature Alignment Loss (MSE between fused features)
                loss_feature = self.mse_loss(student_features, teacher_features.detach())
                
                # 3. Knowledge Distillation Loss (KL-Div on soft logits)
                loss_kd = self._compute_kd_loss(student_logits, teacher_logits.detach())
                
                # Total Loss
                loss_total = loss_ctc + self.alpha * loss_feature + self.beta * loss_kd
            
            # Backward
            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.config.GRAD_CLIP)
            
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_ctc += loss_ctc.item()
            epoch_feat += loss_feature.item()
            epoch_kd += loss_kd.item()
            epoch_total += loss_total.item()
            
            pbar.set_postfix({
                'total': f'{loss_total.item():.3f}',
                'ctc': f'{loss_ctc.item():.3f}',
                'feat': f'{loss_feature.item():.3f}',
                'kd': f'{loss_kd.item():.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            })
        
        n = len(self.train_loader)
        return {
            'total': epoch_total / n,
            'ctc': epoch_ctc / n,
            'feature': epoch_feat / n,
            'kd': epoch_kd / n,
        }

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Validate the Student model on the validation set.
        
        Uses the standard single-input format (LR only).
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}, []
        
        self.student.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []
        
        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        
        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.student(images)  # Standard forward, no return_features
                
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long
                )
                loss = ctc_loss(
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
        
        return {'loss': avg_val_loss, 'acc': val_acc}, submission_data

    def save_model(self, path: str = None) -> None:
        """Save student model checkpoint."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_student_best.pth")
        torch.save(self.student.state_dict(), path)

    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}_student.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def fit(self) -> None:
        """Run the full distillation training loop."""
        print(f"🎓 DISTILLATION START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        print(f"   Alpha (Feature MSE): {self.alpha}")
        print(f"   Beta  (KD KL-Div):   {self.beta}")
        print(f"   Temperature:         {self.temperature}")
        
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            
            # Training with distillation
            train_losses = self.train_one_epoch()
            
            # Validation (Student only, standard LR input)
            val_metrics, submission_data = self.validate()
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"Train [CTC: {train_losses['ctc']:.4f} | "
                  f"Feat: {train_losses['feature']:.4f} | "
                  f"KD: {train_losses['kd']:.4f} | "
                  f"Total: {train_losses['total']:.4f}] | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_student_best.pth")
                print(f"  ⭐ Saved Best Student: {model_path} ({val_acc:.2f}%)")
                
                if submission_data:
                    self.save_submission(submission_data)
        
        # Save final model if no validation
        if self.val_loader is None:
            self.save_model()
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_student_best.pth")
            print(f"  💾 Saved final student: {model_path}")
        
        print(f"\n✅ Distillation complete! Best Val Acc: {self.best_acc:.2f}%")
