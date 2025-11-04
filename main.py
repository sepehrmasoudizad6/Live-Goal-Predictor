"""
Goal Opportunity Detection Model Training System

This module implements a complete training pipeline for detecting goal opportunities
in sports videos using a combination of visual and temporal features. The system uses
a lightweight 3D CNN architecture (TinyVideoNet) with custom temporal feature extraction.

Key Components:
- TinyVideoNet: Efficient 3D CNN for video feature extraction
- FocalLoss: Custom loss function for handling class imbalance
- MultiVideoGoalDataset: Dataset handler for video segments
- Training pipeline with validation and early stopping

Features:
- Mixed precision training support
- Gradient accumulation for larger effective batch sizes
- Class-balanced sampling
- Data augmentation for positive samples
- Early stopping and learning rate scheduling
- Comprehensive metrics tracking

Dependencies:
- PyTorch: Deep learning framework
- OpenCV: Video processing
- scikit-learn: Metrics and evaluation
- NumPy/Pandas: Data handling
- tqdm: Progress tracking

Author: Sepehr Masoudizad
Date: November 4, 2025
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score as sklearn_f1
import os
import glob
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

def get_device():
    """
    Determine the best available compute device for PyTorch.
    
    Returns:
        torch.device: The preferred device in order of:
            1. Apple M1/M2 GPU (MPS) if available
            2. NVIDIA GPU (CUDA) if available
            3. CPU as fallback
    
    The function handles platform-specific availability checks and
    gracefully falls back to CPU if preferred accelerators are
    unavailable or raise exceptions.
    """
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class TinyVideoNet(nn.Module):
    """
    Lightweight 3D CNN for video feature extraction.
    
    This network is designed for efficient processing of video clips,
    using 3D convolutions to capture both spatial and temporal features.
    The architecture progressively reduces spatial and temporal dimensions
    while increasing the feature channels, culminating in a compact
    feature representation.
    
    Architecture:
    - 4 3D convolutional blocks with batch normalization and ReLU
    - Progressive spatial and temporal pooling
    - Final adaptive pooling to fixed size
    - 128-dimensional output feature space
    
    Args:
        pretrained (bool): Currently unused, placeholder for future
                          pretrained model support
    """
    def __init__(self, pretrained=False):
        super(TinyVideoNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.out_features = 128
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification.
    
    This loss function addresses class imbalance by down-weighting
    well-classified examples and focusing on hard, misclassified ones.
    It adds two modulating factors to binary cross-entropy loss:
    - alpha: balances positive vs negative class weights
    - gamma: reduces the relative loss for well-classified examples
    
    Args:
        alpha (float): Weight for positive class (0-1)
        gamma (float): Focusing parameter for modulating factor
        reduction (str): 'mean', 'sum' or 'none' for loss reduction
        
    Reference:
        "Focal Loss for Dense Object Detection" 
        https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_weight * focal_weight * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class GoalOpportunityPredictor(nn.Module):
    """
    Complete model for predicting goal opportunities in video segments.
    
    This model combines visual features from TinyVideoNet with temporal
    features extracted from the video sequence. The architecture uses
    separate projection paths for visual and temporal features before
    combining them for final classification.
    
    Architecture:
    - TinyVideoNet backbone for visual feature extraction
    - Temporal feature projection pathway
    - Feature fusion and classification layers
    
    Args:
        num_frames (int): Number of frames to process (default: 8)
        dropout_rate (float): Dropout probability for regularization
        freeze_backbone (bool): Whether to freeze TinyVideoNet weights
        
    The model produces a single output score indicating the probability
    of a goal opportunity in the given video segment.
    """
    def __init__(self, num_frames=8, dropout_rate=0.3, freeze_backbone=True):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = 256
        self.backbone = TinyVideoNet()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        backbone_out_features = self.backbone.out_features
        self.feature_projection = nn.Sequential(
            nn.Linear(backbone_out_features, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.temporal_projection = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.embed_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.feature_projection, self.temporal_projection, self.classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
    
    def extract_video_features(self, pixel_values):
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            features = self.backbone(pixel_values)
        features = self.feature_projection(features)
        return features
    
    def forward(self, pixel_values, temporal_features):
        video_features = self.extract_video_features(pixel_values)
        temporal_features_proj = self.temporal_projection(temporal_features)
        combined_features = torch.cat([video_features, temporal_features_proj], dim=1)
        output = self.classifier(combined_features)
        return output

class MultiVideoGoalDataset(Dataset):
    """
    Custom Dataset for loading and preprocessing video segments with emotion labels.
    
    This dataset handles loading video segments from multiple source videos,
    extracting frames at specified timestamps, computing temporal features,
    and applying data augmentation for positive samples.
    
    Features:
    - On-the-fly video frame extraction
    - Temporal feature computation
    - Data augmentation (flipping, intensity, contrast)
    - Normalization and preprocessing
    
    Args:
        video_dir (str): Directory containing source videos
        labels_dir (str): Directory containing emotion timeline CSVs
        num_frames (int): Number of frames to extract per segment
        augment (bool): Whether to apply data augmentation
        seed (int): Random seed for reproducibility
        
    Each item consists of:
    - Normalized video frames tensor
    - Computed temporal features
    - Binary label for goal opportunity
    """
    def __init__(self, video_dir, labels_dir, num_frames=8, augment=False, seed=42):
        self.video_dir = video_dir
        self.labels_dir = labels_dir
        self.num_frames = num_frames
        self.augment = augment
        self.seed = seed
        self.mean = torch.tensor([0.45, 0.45, 0.45])
        self.std = torch.tensor([0.225, 0.225, 0.225])
        np.random.seed(seed)
        self.data = self.load_all_videos_and_labels()
    
    def load_all_videos_and_labels(self):
        all_data = []
        csv_files = glob.glob(os.path.join(self.labels_dir, '*_emotion_timeline.csv'))
        for csv_path in csv_files:
            try:
                csv_name = os.path.basename(csv_path)
                video_name = csv_name.replace('_emotion_timeline.csv', '')
                video_path = None
                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    potential_path = os.path.join(self.video_dir, video_name + ext)
                    if os.path.exists(potential_path):
                        video_path = potential_path
                        break
                if video_path is None:
                    continue
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    try:
                        # Parse CSV according to new format
                        start_time = float(row['start_time'])
                        end_time = float(row['end_time'])
                        excited_score = float(row['excited_score'])
                        is_goal = excited_score >= 0.88  # Threshold as per instructions
                        
                        data_item = {
                            'video_path': video_path,
                            'start_time': start_time,
                            'end_time': end_time,
                            'excited_score': excited_score,
                            'is_goal': is_goal
                        }
                        all_data.append(data_item)
                    except Exception:
                        continue
            except Exception:
                continue
        return all_data
    
    def load_video_frames(self, video_path, start_time, end_time):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = max(0, int(start_time * fps))
        end_frame = int(end_time * fps)
        total_available = max(1, end_frame - start_frame)
        frame_indices = np.linspace(start_frame, start_frame + total_available - 1, 
                                   self.num_frames, dtype=int)
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        return frames[:self.num_frames]
    
    def augment_frames(self, frames):
        augmented = []
        for frame in frames:
            if np.random.random() > 0.5:
                frame = cv2.flip(frame, 1)
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.85, 1.15)
                frame = np.clip(frame * factor, 0, 255).astype(np.uint8)
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.9, 1.1)
                mean = frame.mean()
                frame = np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)
            augmented.append(frame)
        return augmented
    
    def normalize_frames(self, frames_tensor):
        mean = self.mean.view(1, 3, 1, 1)
        std = self.std.view(1, 3, 1, 1)
        return (frames_tensor - mean) / std
    
    def compute_temporal_features(self, frames):
        """Compute temporal features directly from video frames"""
        # Duration feature (normalized by 30 seconds)
        duration_val = self.num_frames / 30.0
        
        # Intensity feature (frame differences)
        if len(frames) > 1:
            frames_arr = np.array(frames, dtype=np.float32)
            diffs = np.abs(frames_arr[1:] - frames_arr[:-1])
            intensity_val = np.mean(diffs) / 255.0 * 10.0  # Scale to [0, 10]
        else:
            intensity_val = 0.0
        
        # Excitement feature (frame variance + edge density)
        frames_arr = np.array(frames)
        if len(frames_arr) > 0:
            # Frame variance (std of mean colors)
            frame_means = frames_arr.mean(axis=(1, 2))  # Shape: (num_frames, 3)
            frame_variance = np.std(frame_means) / 255.0
            
            # Edge density on last frame
            last_frame = frames_arr[-1].astype(np.uint8)
            last_gray = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(last_gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            excitement_val = frame_variance * 5.0 + edge_density * 5.0
        else:
            excitement_val = 0.0
        
        return torch.tensor([duration_val, intensity_val, excitement_val], dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        frames = self.load_video_frames(item['video_path'], item['start_time'], item['end_time'])
        if self.augment and item['is_goal']:
            frames = self.augment_frames(frames)
        
        # Compute temporal features directly from frames
        temporal_feats = self.compute_temporal_features(frames)
        
        # Prepare frames tensor
        frames_array = np.stack(frames, axis=0)
        frames_array = frames_array.transpose(0, 3, 1, 2)
        pixel_values = torch.from_numpy(frames_array).float() / 255.0
        pixel_values = self.normalize_frames(pixel_values)
        
        # Label based on excited_score threshold
        label = torch.tensor(1.0 if item['is_goal'] else 0.0, dtype=torch.float32)
        return pixel_values, temporal_feats, label

def train_model(video_dir, labels_dir, epochs=20, batch_size=4,
                use_focal_loss=True, use_weighted_sampler=True,
                augment_data=True, seed=42, accumulation_steps=2):
    """
    Complete training pipeline for the goal opportunity detection model.
    
    This function handles the entire training process including:
    - Dataset preparation and splitting
    - Model initialization and setup
    - Training loop with validation
    - Metrics tracking and model checkpointing
    - Early stopping and learning rate scheduling
    
    Args:
        video_dir (str): Directory containing source videos
        labels_dir (str): Directory containing label CSVs
        epochs (int): Maximum number of training epochs
        batch_size (int): Batch size for training
        use_focal_loss (bool): Whether to use Focal Loss
        use_weighted_sampler (bool): Whether to use class-balanced sampling
        augment_data (bool): Whether to apply data augmentation
        seed (int): Random seed for reproducibility
        accumulation_steps (int): Number of steps for gradient accumulation
        
    Returns:
        GoalOpportunityPredictor: Trained model if successful, None if failed
        
    The function automatically handles device placement, mixed precision
    training, and saves the best model based on validation F1 score.
    """
    device = get_device()
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    full_dataset = MultiVideoGoalDataset(
        video_dir, labels_dir, num_frames=8, augment=False, seed=seed
    )
    if len(full_dataset) == 0:
        return None
    train_size = max(1, int(0.8 * len(full_dataset)))
    val_size = len(full_dataset) - train_size
    indices = list(range(len(full_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_dataset = MultiVideoGoalDataset(
        video_dir, labels_dir, num_frames=8, augment=augment_data, seed=seed
    )
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Weighted sampling for class imbalance
    if use_weighted_sampler:
        train_labels = [full_dataset.data[i]['is_goal'] for i in train_indices]
        class_counts = [train_labels.count(False), train_labels.count(True)]
        if class_counts[0] > 0 and class_counts[1] > 0:
            weights = [1.0/class_counts[0] if not label else 1.0/class_counts[1] 
                      for label in train_labels]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    sampler=sampler, num_workers=0, pin_memory=True)
        else:
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, num_workers=0, pin_memory=True)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                shuffle=True, num_workers=0, pin_memory=True)
    
    val_loader = DataLoader(val_subset, batch_size=batch_size, 
                          shuffle=False, num_workers=0, pin_memory=True)
    
    model = GoalOpportunityPredictor(
        num_frames=8, 
        dropout_rate=0.3, 
        freeze_backbone=False
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
    else:
        pos_weight = torch.tensor([len(train_labels) / max(sum(train_labels), 1)]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-4,
        weight_decay=0.01,
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_f1 = 0.0
    patience_counter = 0
    early_stop_patience = 7
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, unit="batch")
        for batch_idx, (pixel_values, temporal_feats, labels) in enumerate(pbar):
            try:
                pixel_values = pixel_values.to(device, non_blocking=True)
                temporal_feats = temporal_feats.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(pixel_values, temporal_feats)
                        loss = criterion(outputs, labels) / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(pixel_values, temporal_feats)
                    loss = criterion(outputs, labels) / accumulation_steps
                    loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * accumulation_steps
                num_batches += 1
                pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
            except Exception:
                continue
        
        avg_train_loss = train_loss / max(num_batches, 1)
        
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        val_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, unit="batch")
            for batch_idx, (pixel_values, temporal_feats, labels) in enumerate(pbar):
                try:
                    pixel_values = pixel_values.to(device, non_blocking=True)
                    temporal_feats = temporal_feats.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True).unsqueeze(1)
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(pixel_values, temporal_feats)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(pixel_values, temporal_feats)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy().flatten())
                except Exception:
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        
        f1_score = 0.0
        if len(set(all_labels)) > 1 and len(all_labels) > 0:
            try:
                f1_score = sklearn_f1(all_labels, all_preds, pos_label=1.0, zero_division=0)
            except:
                pass
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            if len(set(all_labels)) > 1:
                try:
                    roc_auc = roc_auc_score(all_labels, all_probs)
                except:
                    pass
        
        if f1_score > best_f1:
            best_f1 = f1_score
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'f1_score': f1_score,
            }, 'best_model_fast.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            break
    
    return model

def predict_goal_opportunity(model, video_path, start_time, end_time, 
                           intensity, excited_score, device=None, threshold=0.5, num_frames=8):
    """
    Make predictions on a single video segment.
    
    This function handles the complete prediction pipeline for a single
    video segment, including frame extraction, preprocessing, and model
    inference.
    
    Args:
        model: Trained GoalOpportunityPredictor model
        video_path (str): Path to source video
        start_time (float): Segment start time in seconds
        end_time (float): Segment end time in seconds
        intensity (float): Intensity score from emotion detection
        excited_score (float): Excitement score from emotion detection
        device: Compute device (auto-detected if None)
        threshold (float): Classification threshold
        num_frames (int): Number of frames to extract
        
    Returns:
        tuple: (probability, is_goal)
            - probability (float): Raw model prediction score
            - is_goal (bool): Thresholded binary prediction
            
    The function includes error handling and will return (0.0, False)
    if any step fails during processing.
    """
    if device is None:
        device = get_device()
    model.eval()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0, False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int(start_time * fps))
    end_frame = int(end_time * fps)
    total_available = max(1, end_frame - start_frame)
    frame_indices = np.linspace(start_frame, start_frame + total_available - 1, 
                              num_frames, dtype=int)
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
        ret, frame = cap.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        else:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frames.append(frame)
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    frames = frames[:num_frames]
    
    # Compute temporal features directly from frames
    temporal_feats = MultiVideoGoalDataset.compute_temporal_features(None, frames)
    
    # Prepare frames tensor
    frames_array = np.stack(frames, axis=0).transpose(0, 3, 1, 2)
    pixel_values = torch.from_numpy(frames_array).float() / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
    pixel_values = (pixel_values - mean) / std
    pixel_values = pixel_values.unsqueeze(0).to(device)
    temporal_feats = temporal_feats.unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            output = model(pixel_values, temporal_feats)
            probability = torch.sigmoid(output).item()
            is_goal = probability > threshold
            return probability, is_goal
        except Exception:
            return 0.0, False

if __name__ == "__main__":
    for dir_path in ['videos', 'outputs']:
        if not os.path.exists(dir_path):
            exit(1)
    csv_files = glob.glob('outputs/*_emotion_timeline.csv')
    video_files = glob.glob('videos/*.mp4') + glob.glob('videos/*.avi')
    if not csv_files:
        exit(1)
    if not video_files:
        exit(1)
    model = train_model(
        video_dir='videos',
        labels_dir='outputs',
        epochs=20,
        batch_size=4,
        use_focal_loss=True,
        use_weighted_sampler=True,
        augment_data=True,
        seed=42,
        accumulation_steps=2
    )
    if model is None:
        exit(1)