# Goal Opportunity Detection System

A deep learning-based system for real-time detection of goal opportunities in sports videos using emotion analysis and spatiotemporal features.

## Project Overview

This system implements a complete pipeline for detecting potential goal opportunities in sports videos through:
1. Emotion timeline analysis
2. Real-time video processing
3. Deep learning-based prediction
4. Interactive review and annotation

## System Components

### 1. Emotion Labeling (`labeler.py`)
- Processes video files to generate emotion timelines
- Uses HuBERT model for audio emotion classification
- Generates CSV files with emotion scores and timestamps
- Supports batch processing of multiple videos

### 2. Training System (`main.py`)
- Implements complete training pipeline for goal detection
- Uses TinyVideoNet architecture for efficient video processing
- Features:
  - Mixed precision training
  - Class-balanced sampling
  - Focal Loss for handling imbalanced data
  - Early stopping and learning rate scheduling
  - Comprehensive metrics tracking

### 3. Real-time Predictor (`live_predictor.py`)
- Real-time goal opportunity detection
- Interactive visualization interface
- Features:
  - Live probability tracking
  - Historical trend visualization
  - Performance monitoring (FPS)
  - Adjustable detection threshold
  - Screenshot capture
  - Playback controls

### 4. Review Interface (`review_ui.py`)
- Streamlit-based web interface for reviewing predictions
- Allows manual annotation of goal opportunities
- Features:
  - Video segment playback
  - Emotion timeline filtering
  - Batch statistics
  - Export functionality

## Directory Structure

```
.
├── labeler.py           # Emotion timeline generation
├── main.py             # Training pipeline
├── live_predictor.py   # Real-time detection
├── review_ui.py        # Review interface
├── best_model.pth      # Trained model weights
├── best_model_fast.pth # Optimized model weights
├── videos/             # Source video files
├── outputs/            # Generated emotion timelines
└── labels/            # Annotation data
```

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch
- torchaudio
- OpenCV
- Streamlit
- transformers (HuggingFace)
- ffmpeg-python
- pandas
- numpy
- scikit-learn

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg (if not already installed)
# macOS:
brew install ffmpeg
# Ubuntu:
sudo apt-get install ffmpeg
```

## Usage

### 1. Generate Emotion Timelines
```bash
python labeler.py
```
This will process all videos in the `videos/` directory and generate emotion timeline CSVs in `outputs/`.

### 2. Train the Model
```bash
python main.py
```
Trains the goal detection model using the generated emotion timelines and video data.

### 3. Run Real-time Detection
```bash
python live_predictor.py
```
Interactive interface for real-time goal opportunity detection.

Controls:
- 'q': Quit
- 's': Save screenshot
- 'r': Reset prediction history
- 'p': Pause/resume playback
- '+'/'-': Adjust detection threshold

### 4. Review and Annotate
```bash
streamlit run review_ui.py
```
Opens the web interface for reviewing and annotating detected segments.

## Model Architecture

### TinyVideoNet
- Lightweight 3D CNN for video feature extraction
- Progressive spatial and temporal pooling
- 128-dimensional feature space
- Optimized for real-time inference

### Goal Opportunity Predictor
- Combined visual and temporal feature processing
- Separate projection pathways
- Dropout regularization
- Binary classification output

## Performance Metrics

The system is evaluated on:
- F1 Score
- ROC-AUC
- Precision-Recall curves
- Real-time FPS
- Inference latency

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Authors

Sepehr Masoudizad

## Acknowledgments

- HuggingFace Transformers library
- PyTorch team
- Streamlit framework