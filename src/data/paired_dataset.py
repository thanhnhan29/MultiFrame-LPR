"""Paired HR/LR MultiFrameDataset for Teacher-Student Knowledge Distillation."""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_train_transforms,
    get_val_transforms,
    get_light_transforms,
)


class PairedMultiFrameDataset(Dataset):
    """Dataset that returns PAIRED (LR, HR) frames for each track.
    
    Used for Knowledge Distillation training:
    - LR images -> Student model (with augmentation)
    - HR images -> Teacher model (clean transform only)
    
    Only includes tracks that have BOTH lr-*.png and hr-*.png files.
    """
    
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        full_train: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            img_height: Target image height.
            img_width: Target image width.
            char2idx: Character to index mapping.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            augmentation_level: 'full' or 'light' augmentation for LR training images.
            full_train: If True, use all tracks for training (no val split).
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.full_train = full_train
        
        # LR transform: augmentation for training, clean for val
        if mode == 'train':
            if augmentation_level == "light":
                self.lr_transform = get_light_transforms(img_height, img_width)
            else:
                self.lr_transform = get_train_transforms(img_height, img_width)
        else:
            self.lr_transform = get_val_transforms(img_height, img_width)
        
        # HR transform: always clean (Teacher needs clean input)
        self.hr_transform = get_val_transforms(img_height, img_width)

        print(f"[{mode.upper()} PAIRED] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
        
        selected_tracks = train_tracks if mode == 'train' else val_tracks
        print(f"[{mode.upper()} PAIRED] Loaded {len(selected_tracks)} tracks.")
        
        self._index_paired_samples(selected_tracks)
        print(f"-> Total: {len(self.samples)} paired samples.")

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one with Scenario-B priority.
        
        Reuses the same split file as the original dataset for consistency.
        """
        if self.full_train:
            print("📌 FULL TRAIN MODE: Using all tracks for training (no validation split).")
            return all_tracks, []
        
        train_tracks, val_tracks = [], []
        
        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, 'r') as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            
            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                print("⚠️ Split file invalid or missing Scenario-B. Recreating...")
                val_tracks = []

        if not val_tracks:
            print("⚠️ Creating new split (Taking Val only from Scenario-B)...")
            
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t]
            
            if not scenario_b_tracks:
                print("⚠️ Warning: No 'Scenario-B' folder found. Using random from all.")
                scenario_b_tracks = all_tracks
            
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]
            
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_paired_samples(self, tracks: List[str]) -> None:
        """Index only tracks that contain BOTH LR and HR images."""
        skipped = 0
        for track_path in tqdm(tracks, desc=f"Indexing paired {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label:
                    continue
                
                track_id = os.path.basename(track_path)
                
                lr_files = sorted(
                    glob.glob(os.path.join(track_path, "lr-*.png")) +
                    glob.glob(os.path.join(track_path, "lr-*.jpg"))
                )
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png")) +
                    glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )
                
                # Only include if BOTH LR and HR exist
                if lr_files and hr_files:
                    self.samples.append({
                        'lr_paths': lr_files,
                        'hr_paths': hr_files,
                        'label': label,
                        'track_id': track_id,
                    })
                else:
                    skipped += 1
            except Exception:
                pass
        
        if skipped > 0:
            print(f"⚠️ Skipped {skipped} tracks without both LR and HR images.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, str, str]:
        """Load paired (LR, HR) frames for one track.
        
        Returns:
            lr_images: [5, 3, H, W] - LR frames (augmented for train)
            hr_images: [5, 3, H, W] - HR frames (clean transform)
            target: [target_len] - encoded label
            target_len: int - length of target
            label: str - raw label text
            track_id: str - track identifier
        """
        item = self.samples[idx]
        lr_paths = item['lr_paths']
        hr_paths = item['hr_paths']
        label = item['label']
        track_id = item['track_id']
        
        # Load LR frames
        lr_images = []
        for p in lr_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.lr_transform(image=image)['image']
            lr_images.append(image)
        lr_tensor = torch.stack(lr_images, dim=0)
        
        # Load HR frames
        hr_images = []
        for p in hr_paths:
            image = cv2.imread(p, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.hr_transform(image=image)['image']
            hr_images.append(image)
        hr_tensor = torch.stack(hr_images, dim=0)
        
        # Encode label
        target = [self.char2idx[c] for c in label if c in self.char2idx]
        if len(target) == 0:
            target = [0]
        target_len = len(target)
            
        return lr_tensor, hr_tensor, torch.tensor(target, dtype=torch.long), target_len, label, track_id

    @staticmethod
    def collate_fn(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        """Custom collate function for paired DataLoader."""
        lr_images, hr_images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        lr_images = torch.stack(lr_images, 0)
        hr_images = torch.stack(hr_images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return lr_images, hr_images, targets, target_lengths, labels_text, track_ids
