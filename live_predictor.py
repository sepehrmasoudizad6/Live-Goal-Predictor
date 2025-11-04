"""
Real-time Goal Opportunity Detection System

This module implements a real-time video analysis system for detecting potential
goal opportunities in sports footage. It uses a lightweight neural network model
to process video frames and provides immediate visual feedback through an
interactive display interface.

Key Features:
- Real-time frame processing and prediction
- Live visualization with probability tracking
- Interactive threshold adjustment
- Performance monitoring (FPS, buffer status)
- Screenshot capture and playback controls
- Trend visualization with historical predictions

Dependencies:
- PyTorch: Neural network inference
- OpenCV: Video capture and display
- NumPy: Numerical computations
- Logging: System status tracking

Author: Sepehr Masoudizad
Date: November 4, 2025
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyVideoNet(nn.Module):
    """
    Lightweight 3D CNN for efficient video feature extraction.
    
    This network is designed for real-time processing of video segments,
    using 3D convolutions to capture spatiotemporal features. The architecture
    progressively reduces spatial and temporal dimensions while increasing the
    feature channels, optimized for low-latency inference.
    
    Architecture:
    - Input: Video clip of shape (N, C, T, H, W)
    - 4 3D convolutional blocks with batch normalization
    - Progressive spatial and temporal pooling
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

class GoalOpportunityPredictor(nn.Module):
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

class LiveGoalPredictor:
    """
    Real-time goal opportunity detection system.
    
    This class manages the real-time processing pipeline for detecting goal
    opportunities in a video stream. It maintains a frame buffer for temporal
    analysis and provides frame-by-frame predictions using the provided model.
    
    Features:
    - Frame buffer management for temporal analysis
    - Real-time frame preprocessing and normalization
    - Temporal feature extraction
    - Prediction history tracking
    - Performance monitoring (FPS)
    
    Args:
        model: Trained GoalOpportunityPredictor model
        threshold (float): Decision threshold for goal opportunity detection
    """
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.device = next(model.parameters()).device
        self.threshold = threshold
        self.num_frames = model.num_frames
        self.frame_buffer = deque(maxlen=self.num_frames)
        self.prediction_history = deque(maxlen=100)
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        self.current_probability = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        logger.info(f"Predictor initialized on {self.device}")
        logger.info(f"Using {self.num_frames} frames per prediction")
    
    def preprocess_frame(self, frame):
        """
        Preprocess a single video frame for model input.
        
        Steps:
        1. Convert BGR to RGB color space
        2. Resize to model input size (224x224)
        3. Convert to float32 for numerical stability
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            numpy.ndarray: Preprocessed frame ready for batching
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (224, 224), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32)
    
    def normalize_frames(self, frames_tensor):
        mean = self.mean.to(frames_tensor.device)
        std = self.std.to(frames_tensor.device)
        return (frames_tensor - mean) / std
    
    def calculate_temporal_features(self):
        """
        Calculate temporal features from the frame buffer.
        
        Extracts three key temporal features:
        1. Duration: Normalized segment length
        2. Intensity: Average frame-to-frame difference
        3. Excitement: Combination of motion variance and edge density
        
        The features capture different aspects of temporal dynamics:
        - Frame-to-frame changes (motion intensity)
        - Overall scene complexity (edge density)
        - Visual variance over time
        
        Returns:
            torch.Tensor: Feature vector [duration, intensity, excitement]
                         on the appropriate device
        """
        if len(self.frame_buffer) < self.num_frames:
            return torch.zeros(1, 3, device=self.device, dtype=torch.float32)
        
        frames = np.array(list(self.frame_buffer), dtype=np.float32)
        
        # Duration feature (fixed based on frame count)
        duration_val = self.num_frames / 30.0
        
        # Intensity feature (frame differences)
        diffs = np.abs(frames[1:] - frames[:-1])
        intensity_val = np.mean(diffs) / 255.0 * 10.0
        
        # Excitement feature
        frame_means = frames.mean(axis=(1, 2))  # Shape: (num_frames, 3)
        frame_variance = np.std(frame_means) / 255.0
        
        last_frame = frames[-1].astype(np.uint8)
        last_gray = cv2.cvtColor(last_frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(last_gray, 100, 200)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        excitement_val = frame_variance * 5.0 + edge_density * 5.0
        
        return torch.tensor(
            [[duration_val, intensity_val, excitement_val]], 
            device=self.device, 
            dtype=torch.float32
        )
    
    def add_frame_and_predict(self, frame):
        """
        Process a new frame and generate prediction.
        
        This method implements the complete frame processing pipeline:
        1. Preprocess and add frame to buffer
        2. Extract frames tensor and temporal features
        3. Run model inference
        4. Update prediction history and performance metrics
        
        Args:
            frame: Raw input frame in BGR format
            
        Returns:
            float: Prediction probability [0-1] for goal opportunity
            
        Performance metrics (FPS, frame count) are updated automatically.
        """
        start_time = time.time()
        self.frame_buffer.append(self.preprocess_frame(frame))
        self.frame_count += 1
        
        if len(self.frame_buffer) < self.num_frames:
            return self.current_probability
        
        # Prepare frames tensor
        frames_array = np.array(list(self.frame_buffer))
        frames_array = frames_array.transpose(0, 3, 1, 2)
        pixel_values = torch.from_numpy(frames_array).float() / 255.0
        pixel_values = self.normalize_frames(pixel_values)
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        
        # Compute temporal features
        temporal_features = self.calculate_temporal_features()
        
        with torch.inference_mode():
            logits = self.model(pixel_values, temporal_features)
            probability = torch.sigmoid(logits).item()
        
        self.current_probability = probability
        self.prediction_history.append(probability)
        elapsed = time.time() - start_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0.0
        return probability

def create_video_display(frame, predictor, threshold, target_width=1200):
    """
    Create an interactive visualization of the prediction results.
    
    This function generates a rich visual display combining the video frame
    with real-time prediction information, including:
    - Current prediction probability with status
    - Probability bar with threshold marker
    - Historical trend graph
    - Performance metrics (FPS, buffer status)
    - Visual alerts for goal opportunities
    
    Args:
        frame: Input video frame
        predictor: LiveGoalPredictor instance
        threshold: Current decision threshold
        target_width: Desired display width (maintains aspect ratio)
        
    Returns:
        numpy.ndarray: Composed display frame with overlays
    """
    h, w = frame.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    display_frame = cv2.resize(frame, (target_width, new_h))
    overlay = display_frame.copy()
    prob = predictor.current_probability
    prob_pct = prob * 100
    if prob >= threshold:
        color = (0, 255, 0)
        status = "GOAL OPPORTUNITY!"
    elif prob >= threshold * 0.7:
        color = (0, 255, 255)
        status = "High Probability"
    else:
        color = (255, 100, 100)
        status = "Monitoring"
    bar_height = 120
    cv2.rectangle(overlay, (0, 0), (target_width, bar_height), (0, 0, 0), -1)
    cv2.putText(overlay, status, (30, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1.4, color, 4)
    cv2.putText(overlay, "[INVERTED]", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
    prob_text = f"{prob_pct:.1f}%"
    cv2.putText(overlay, prob_text, (target_width - 250, 65),
                cv2.FONT_HERSHEY_COMPLEX, 1.8, (255, 255, 255), 4)
    fps_text = f"FPS: {predictor.fps:.1f}"
    cv2.putText(overlay, fps_text, (target_width - 250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    bar_y = bar_height + 20
    bar_width = target_width - 60
    bar_x = 30
    bar_h = 40
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h),
                  (50, 50, 50), -1)
    threshold_x = bar_x + int(bar_width * threshold)
    cv2.line(overlay, (threshold_x, bar_y - 10), (threshold_x, bar_y + bar_h + 10),
             (255, 255, 0), 4)
    cv2.putText(overlay, "T", (threshold_x - 12, bar_y - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    fill_width = int(bar_width * prob)
    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_h),
                  color, -1)
    for i in range(0, 101, 25):
        x = bar_x + int(bar_width * i / 100)
        cv2.line(overlay, (x, bar_y + bar_h), (x, bar_y + bar_h + 8),
                 (150, 150, 150), 2)
        cv2.putText(overlay, f"{i}%", (x - 20, bar_y + bar_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    if len(predictor.prediction_history) > 1:
        trend_height = 100
        trend_margin = 15
        trend_y_start = new_h - trend_height - trend_margin
        trend_bg = (30, 30, 30)
        cv2.rectangle(overlay, (bar_x, trend_y_start),
                      (bar_x + bar_width, trend_y_start + trend_height),
                      trend_bg, -1)
        history = list(predictor.prediction_history)
        points = []
        max_points = min(len(history), 50)
        start_idx = len(history) - max_points
        for i, prob_val in enumerate(history[start_idx:]):
            x = bar_x + int((i / max_points) * bar_width)
            y = trend_y_start + trend_height - int(prob_val * trend_height)
            points.append((x, y))
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt_color = (0, 255, 0) if history[start_idx + i] >= threshold else (100, 149, 237)
                cv2.line(overlay, points[i], points[i + 1], pt_color, 3)
            for i, prob_val in enumerate(history[start_idx:]):
                if prob_val >= threshold:
                    cv2.circle(overlay, points[i], 5, (0, 255, 0), -1)
        threshold_y = trend_y_start + trend_height - int(threshold * trend_height)
        cv2.line(overlay, (bar_x, threshold_y), (bar_x + bar_width, threshold_y),
                 (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, "Probability Trend (TinyVideoNet)", (bar_x, trend_y_start - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    buffer_status = f"Buffer: {len(predictor.frame_buffer)}/{predictor.num_frames}"
    cv2.putText(overlay, buffer_status, (30, bar_y + bar_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    if prob >= threshold:
        pulse = int(abs(np.sin(time.time() * 10)) * 15) + 8
        cv2.rectangle(overlay, (0, 0), (target_width, new_h),
                      (0, 255, 0), pulse)
        marker_size = 70
        thickness = 8
        cv2.line(overlay, (0, 0), (marker_size, 0), (0, 255, 0), thickness)
        cv2.line(overlay, (0, 0), (0, marker_size), (0, 255, 0), thickness)
        cv2.line(overlay, (target_width, 0), (target_width - marker_size, 0), (0, 255, 0), thickness)
        cv2.line(overlay, (target_width, 0), (target_width, marker_size), (0, 255, 0), thickness)
        cv2.line(overlay, (0, new_h), (marker_size, new_h), (0, 255, 0), thickness)
        cv2.line(overlay, (0, new_h), (0, new_h - marker_size), (0, 255, 0), thickness)
        cv2.line(overlay, (target_width, new_h), (target_width - marker_size, new_h), (0, 255, 0), thickness)
        cv2.line(overlay, (target_width, new_h), (target_width, new_h - marker_size), (0, 255, 0), thickness)
    alpha = 0.85
    display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
    return display_frame

def load_model(model_path, threshold=0.5):
    """
    Load and initialize the goal detection model.
    
    This function handles:
    - Device selection (MPS, CUDA, or CPU)
    - Model initialization and weight loading
    - Architecture compatibility verification
    - Configuration of the prediction system
    
    Args:
        model_path (str): Path to the saved model checkpoint
        threshold (float): Initial decision threshold
        
    Returns:
        LiveGoalPredictor: Initialized prediction system
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model architecture doesn't match checkpoint
    """
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU - performance will be limited")
    
    logger.info("Creating lightweight TinyVideoNet model (256-dim embeddings)...")
    model = GoalOpportunityPredictor(
        num_frames=8,
        dropout_rate=0.3,
        freeze_backbone=False
    )
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'classifier.0.weight' in state_dict:
            checkpoint_shape = state_dict['classifier.0.weight'].shape
            logger.info(f"✓ Checkpoint classifier input: {checkpoint_shape[1]} dims")
            logger.info(f"✓ Model classifier input: {model.classifier[0].in_features} dims")
            if checkpoint_shape[1] != model.classifier[0].in_features:
                logger.error("❌ Architecture mismatch detected!")
                raise ValueError(
                    f"Model architecture mismatch!\n"
                    f"Checkpoint expects {checkpoint_shape[1]} dims, "
                    f"but model has {model.classifier[0].in_features} dims.\n"
                    f"Make sure live_predictor.py matches train.py architecture."
                )
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✓ Model weights loaded successfully!")
    model.to(device)
    model.eval()
    logger.info(f"Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best F1 Score: {checkpoint.get('f1_score', 'unknown'):.4f}")
    return LiveGoalPredictor(model, threshold)

def main():
    """
    Main entry point for the real-time goal detection system.
    
    This function implements the complete runtime loop including:
    - Model loading and initialization
    - Video capture setup
    - Real-time frame processing
    - Interactive display and controls
    - User input handling
    
    Controls:
    - 'q': Quit the application
    - 's': Save screenshot
    - 'r': Reset prediction history
    - 'p': Pause/resume playback
    - '+'/'-': Adjust detection threshold
    
    Configuration:
    - MODEL_PATH: Path to trained model weights
    - THRESHOLD: Initial detection threshold
    - VIDEO_SOURCE: Input video file or camera index
    - FRAME_SKIP: Process every Nth frame
    - VIDEO_WIDTH: Display window width
    """
    MODEL_PATH = "best_model_fast.pth"
    THRESHOLD = 0.5  # This is the model output threshold, not the excited_score threshold
    VIDEO_SOURCE = '2.mp4'
    FRAME_SKIP = 4
    VIDEO_WIDTH = 1200
    
    try:
        predictor = load_model(MODEL_PATH, THRESHOLD)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {VIDEO_SOURCE}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")
    
    cv2.namedWindow('Goal Opportunity Predictor', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Goal Opportunity Predictor', VIDEO_WIDTH, int(VIDEO_WIDTH * height / width) + 250)
    
    frame_count = 0
    threshold = THRESHOLD
    paused = False
    logger.info("Starting real-time prediction...")
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("End of video or capture failed")
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP == 0:
                    probability = predictor.add_frame_and_predict(frame)
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    status = f"Frame {predictor.frame_count:5d} ({progress:5.1f}%) | " \
                            f"Probability: {probability*100:5.1f}% | " \
                            f"FPS: {predictor.fps:4.1f}"
                    if probability >= threshold:
                        status += " | 🚀 GOAL OPPORTUNITY DETECTED!"
            else:
                # Use last frame when paused
                pass
            
            video_display = create_video_display(frame, predictor, threshold, VIDEO_WIDTH)
            if paused:
                cv2.putText(video_display, "PAUSED", (VIDEO_WIDTH // 2 - 100, 150),
                           cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 255, 255), 5)
            cv2.imshow('Goal Opportunity Predictor', video_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                cv2.imwrite(filename, video_display)
                logger.info(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                predictor.prediction_history.clear()
                predictor.frame_buffer.clear()
                logger.info("History and buffer reset")
            elif key == ord('p'):
                paused = not paused
                logger.info(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('+') or key == ord('='):
                threshold = min(1.0, threshold + 0.05)
                logger.info(f"Threshold increased to {threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                threshold = max(0.0, threshold - 0.05)
                logger.info(f"Threshold decreased to {threshold:.2f}")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()