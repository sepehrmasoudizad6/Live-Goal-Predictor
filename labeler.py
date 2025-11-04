"""
Audio Emotion Labeling System

This script implements an automated system for labeling emotions in video content by analyzing
the audio track. It uses a pre-trained HuBERT model to classify audio segments into different
emotional categories, primarily focusing on 'excited' vs 'neutral' states.

Key Features:
- Extracts audio from video files using ffmpeg
- Processes audio in sliding windows for continuous emotion analysis
- Classifies emotions using the SUPERB HuBERT model
- Generates timeline CSV files with emotion labels and confidence scores
- Supports batch processing of multiple video files

Dependencies:
- torch: For deep learning model operations
- torchaudio: For audio processing
- transformers: For the HuBERT model
- ffmpeg-python: For video/audio manipulation
- pandas: For data handling and CSV output

Author: Sepehr Masoudizad
Date: November 4, 2025
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import torch
import torchaudio
import ffmpeg
import pandas as pd
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


@dataclass
class ThresholdConfig:
    """Configuration for emotion classification thresholds.
    
    Attributes:
        excited_intensity_threshold (float): Minimum score needed to classify as 'excited'
        neutral_threshold (float): Minimum score needed for high-confidence 'neutral'
    """
    excited_intensity_threshold: float = 0.91
    neutral_threshold: float = 0.60


# Emotion pattern sets for classification
EXCITED_PATTERNS = {"hap", "ang", "exc"}  # happiness, anger, excitement
NEUTRAL_PATTERNS = {"neu", "sad", "cal", "dis"}  # neutral, sadness, calm, disgust
AMBIGUOUS_PATTERNS = {"sur", "fea"}  # surprise, fear - treated as excited with lower confidence


def categorize_label(label: str) -> tuple[str, float]:
    """Categorize a label and return (category, confidence_weight)."""
    label_lower = label.lower()
    
    for pattern in EXCITED_PATTERNS:
        if pattern in label_lower:
            return "excited", 1.0
    
    for pattern in NEUTRAL_PATTERNS:
        if pattern in label_lower:
            return "neutral", 1.0
    
    for pattern in AMBIGUOUS_PATTERNS:
        if pattern in label_lower:
            return "excited", 0.7
    
    return "neutral", 0.5


def classify_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    feature_extractor: AutoFeatureExtractor,
    model: AutoModelForAudioClassification,
    device: torch.device,
    thresholds: ThresholdConfig,
    window_sec: float,
    step_sec: float,
    debug: bool = False,
) -> list[dict]:
    """Analyze audio segments using sliding windows and classify emotions.
    
    Args:
        waveform (torch.Tensor): Input audio waveform
        sample_rate (int): Audio sampling rate in Hz
        feature_extractor: HuBERT feature extractor
        model: Pre-trained emotion classification model
        device: Torch device for computation
        thresholds: Configuration for classification thresholds
        window_sec (float): Duration of each analysis window in seconds
        step_sec (float): Step size between windows in seconds
        debug (bool): Enable detailed debug output
        
    Returns:
        list[dict]: Timeline of emotion classifications with metadata:
            - start_time: Start of the segment in seconds
            - end_time: End of the segment in seconds
            - emotion: 'excited' or 'neutral'
            - intensity: Confidence score for the classification
            - confidence: 'high' or 'low'
            - excited_score: Raw excited category score
            - neutral_score: Raw neutral category score
    """
    data = waveform.squeeze().numpy()
    window_size = int(window_sec * sample_rate)
    step_size = int(step_sec * sample_rate)

    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window and step sizes must be positive.")
    if len(data) < window_size:
        return []

    if debug:
        print("\n[DEBUG] Model labels with categorization:")
        for idx, label in model.config.id2label.items():
            category, weight = categorize_label(label)
            print(f"  {idx}: '{label}' -> {category} (weight={weight:.1f})")
        print()

    timeline: list[dict] = []
    segment_count = 0
    
    for start_idx in range(0, len(data) - window_size + 1, step_size):
        segment = data[start_idx : start_idx + window_size]
        start_time = start_idx / sample_rate
        end_time = (start_idx + window_size) / sample_rate

        inputs = feature_extractor(
            segment,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        tensor_inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

        with torch.no_grad():
            logits = model(**tensor_inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()

        # Calculate weighted scores by category
        excited_score = 0.0
        neutral_score = 0.0
        
        for idx, prob in enumerate(probs):
            label = model.config.id2label[idx]
            category, weight = categorize_label(label)
            weighted_prob = float(prob) * weight
            
            if category == "excited":
                excited_score += weighted_prob
            else:
                neutral_score += weighted_prob
        
        # Normalize scores
        total_score = excited_score + neutral_score
        if total_score > 0:
            excited_score /= total_score
            neutral_score /= total_score
        
        if debug and segment_count < 3:
            print(f"[DEBUG] Segment {segment_count} ({start_time:.2f}s - {end_time:.2f}s):")
            print(f"  Excited score: {excited_score:.3f}")
            print(f"  Neutral score: {neutral_score:.3f}")
        
        # SIMPLE DECISION LOGIC: excited if intensity > 0.85
        if excited_score >= thresholds.excited_intensity_threshold:
            emotion = "excited"
            intensity = round(excited_score, 3)
            confidence = "high"
        elif neutral_score >= thresholds.neutral_threshold:
            emotion = "neutral"
            intensity = round(neutral_score, 3)
            confidence = "high"
        else:
            emotion = "neutral"
            intensity = round(max(neutral_score, excited_score), 3)
            confidence = "low"
        
        if debug and segment_count < 3:
            print(f"  -> Decision: {emotion} (intensity={intensity}, confidence={confidence})\n")
        
        timeline.append(
            {
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "emotion": emotion,
                "intensity": intensity,
                "confidence": confidence,
                "excited_score": round(excited_score, 3),
                "neutral_score": round(neutral_score, 3),
            }
        )
        segment_count += 1

    return timeline


def process_video(
    video_path: Path,
    output_dir: Path,
    feature_extractor: AutoFeatureExtractor,
    model: AutoModelForAudioClassification,
    device: torch.device,
    thresholds: ThresholdConfig,
    window_sec: float,
    step_sec: float,
    debug: bool = False,
) -> Path | None:
    """Process a single video file through the complete emotion analysis pipeline.
    
    This function handles the entire workflow for a single video:
    1. Extracts audio track from video
    2. Resamples audio to target sample rate if needed
    3. Performs emotion classification on audio segments
    4. Saves results to CSV with detailed statistics
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save CSV results
        feature_extractor: HuBERT feature extractor
        model: Pre-trained emotion classification model
        device: Torch device for computation
        thresholds: Classification threshold configuration
        window_sec: Duration of analysis windows
        step_sec: Step size between windows
        debug: Enable detailed debug output
        
    Returns:
        Path: Path to output CSV file if successful, None if failed
        
    The output CSV contains a timeline of emotion classifications with
    metadata including start/end times, emotion labels, and confidence scores.
    """
    tmp_audio = None
    try:
        tmp_audio = extract_audio(video_path)
        waveform, sr = torchaudio.load(tmp_audio)
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE

        timeline = classify_segments(
            waveform=waveform,
            sample_rate=sr,
            feature_extractor=feature_extractor,
            model=model,
            device=device,
            thresholds=thresholds,
            window_sec=window_sec,
            step_sec=step_sec,
            debug=debug,
        )

        if not timeline:
            print(f"[skip] {video_path.name}: audio shorter than window; no output.")
            return None

        df = pd.DataFrame(timeline)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{video_path.stem}_emotion_timeline.csv"
        df.to_csv(csv_path, index=False)
        print(f"[ok] {video_path.name}: wrote {csv_path.name}")
        
        # Print emotion distribution
        emotion_counts = df['emotion'].value_counts()
        confidence_counts = df['confidence'].value_counts()
        high_confidence_excited = len(df[(df['emotion'] == 'excited') & (df['confidence'] == 'high')])
        
        print(f"     Emotion distribution: {dict(emotion_counts)}")
        print(f"     Confidence levels: {dict(confidence_counts)}")
        print(f"     High-confidence excited segments: {high_confidence_excited}")
        
        return csv_path

    except Exception as exc:
        print(f"[error] {video_path.name}: {exc}")
        return None
    finally:
        if tmp_audio and tmp_audio.exists():
            tmp_audio.unlink(missing_ok=True)


# Supported video formats and audio configuration
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}
TARGET_SAMPLE_RATE = 16_000  # Hz, required by HuBERT model


def extract_audio(video_path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Extract audio track from video file as mono WAV.
    
    Args:
        video_path: Input video file path
        sample_rate: Target audio sample rate in Hz
        
    Returns:
        Path: Temporary WAV file path
        
    Uses ffmpeg to extract audio track, convert to mono,
    and resample to the target sample rate. The output is
    saved as a temporary WAV file that should be cleaned up
    after use.
    """
    tmp_file = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    (
        ffmpeg.input(str(video_path))
        .output(str(tmp_file), format="wav", ac=1, ar=sample_rate)
        .overwrite_output()
        .run(quiet=True)
    )
    return tmp_file


def find_videos(videos_dir: Path) -> list[Path]:
    """Return all video files in the directory matching the allowed extensions."""
    return sorted(
        file
        for file in videos_dir.iterdir()
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS
    )


@dataclass
class Config:
    """Configuration for the emotion labeling pipeline.
    
    Attributes:
        videos_dir: Directory containing input video files
        output_dir: Directory for CSV output files
        window_sec: Duration of analysis windows (seconds)
        step_sec: Step size between windows (seconds)
        excited_intensity_threshold: Threshold for excited classification
        neutral_threshold: Threshold for neutral classification
        model_name: HuggingFace model identifier
        debug: Enable detailed debug output
        
    The window_sec and step_sec parameters control the granularity
    of the analysis. Smaller values give more detail but increase
    processing time. The step_sec should be <= window_sec to ensure
    continuous coverage.
    """
    videos_dir: Path = Path("videos")
    output_dir: Path = Path("outputs")
    window_sec: float = 3.0
    step_sec: float = 2.0
    excited_intensity_threshold: float = 0.85
    neutral_threshold: float = 0.60
    model_name: str = "superb/hubert-base-superb-er"
    debug: bool = True


def main(config: Config | None = None) -> None:
    """Main entry point for the emotion labeling pipeline.
    
    This function orchestrates the complete workflow:
    1. Validates input/output directories
    2. Loads ML model and moves it to appropriate device
    3. Processes all videos in the input directory
    4. Generates summary statistics for the batch
    
    Args:
        config: Optional configuration override. If None,
               default configuration will be used.
               
    Raises:
        FileNotFoundError: If videos directory doesn't exist
                          or contains no supported video files
    
    The function processes all supported video files in the
    input directory and generates individual CSV timelines
    for each, along with batch statistics for monitoring.
    """
    if config is None:
        config = Config()

    videos_dir: Path = config.videos_dir
    output_dir: Path = config.output_dir

    if not videos_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {videos_dir}")

    video_files = find_videos(videos_dir)
    if not video_files:
        raise FileNotFoundError(f"No supported video files in {videos_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[info] Loading model '{config.model_name}' on {device}…")
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
    model = AutoModelForAudioClassification.from_pretrained(config.model_name).to(device)
    model.eval()

    thresholds = ThresholdConfig(
        excited_intensity_threshold=config.excited_intensity_threshold,
        neutral_threshold=config.neutral_threshold,
    )

    successes = []
    failures = []

    for idx, video_path in enumerate(video_files):
        result = process_video(
            video_path=video_path,
            output_dir=output_dir,
            feature_extractor=feature_extractor,
            model=model,
            device=device,
            thresholds=thresholds,
            window_sec=config.window_sec,
            step_sec=config.step_sec,
            debug=(config.debug and idx == 0),
        )
        if result:
            successes.append(video_path.name)
        else:
            failures.append(video_path.name)

    print("\n=== Batch summary ===")
    print(f"Processed: {len(video_files)} total")
    print(f"Success:   {len(successes)}")
    print(f"Failed:    {len(failures)}")

    if successes:
        print("Successful files:")
        for name in successes:
            print(f"  - {name}")
    if failures:
        print("Failed files:")
        for name in failures:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
