"""
Video Emotion Review Tool

A Streamlit-based web interface for reviewing and annotating emotion segments
in video content, specifically focused on identifying goal opportunities in
sports videos. This tool allows human reviewers to validate and annotate
the automated emotion detection results.

Key Features:
- Interactive video segment review
- Filterable timeline visualization
- Annotation of goal opportunities
- Statistics tracking and reporting
- Export functionality for reviewed data
- Note-taking for each segment

Dependencies:
- streamlit: Web interface framework
- pandas: Data management
- pathlib: File system operations

Author: Sepehr Masoudizad
Date: November 4, 2025
"""

import streamlit as st
from pathlib import Path
import pandas as pd
from typing import Optional

st.set_page_config(page_title="Video Emotion Review", layout="wide")

# Application Constants
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}  # Supported video formats
OUTPUTS_DIR = Path("outputs")  # Directory containing emotion timeline CSVs
VIDEOS_DIR = Path("videos")    # Directory containing source videos


def get_available_timelines() -> dict[str, Path]:
    """
    Scan the outputs directory for emotion timeline CSV files.
    
    Returns:
        dict[str, Path]: Mapping of video names (without extension) to their
                        corresponding timeline CSV file paths.
    
    The function looks for files ending in '_emotion_timeline.csv' and creates
    a mapping using the base video name as the key. This allows easy lookup
    of timeline data for each video file.
    """
    if not OUTPUTS_DIR.exists():
        return {}
    
    timelines = {}
    for csv_file in OUTPUTS_DIR.glob("*_emotion_timeline.csv"):
        video_stem = csv_file.stem.replace("_emotion_timeline", "")
        timelines[video_stem] = csv_file
    
    return timelines


def find_video_file(video_stem: str) -> Optional[Path]:
    """
    Locate the video file corresponding to a timeline.
    
    Args:
        video_stem (str): Base name of the video without extension
        
    Returns:
        Optional[Path]: Path to the video file if found, None otherwise
        
    The function tries all supported video extensions (mp4, mkv, etc.)
    to find the matching video file in the videos directory.
    """
    if not VIDEOS_DIR.exists():
        return None
    
    for ext in VIDEO_EXTENSIONS:
        video_path = VIDEOS_DIR / f"{video_stem}{ext}"
        if video_path.exists():
            return video_path
    return None


def load_timeline(csv_path: Path) -> pd.DataFrame:
    """
    Load and prepare the emotion timeline data for review.
    
    Args:
        csv_path (Path): Path to the timeline CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the timeline data with
                     additional columns for review annotations
                     
    The function ensures the DataFrame has the required columns for
    review annotations (is_goal_opportunity and notes), adding them
    if they don't exist in the original data.
    """
    df = pd.read_csv(csv_path)
    
    if "is_goal_opportunity" not in df.columns:
        df["is_goal_opportunity"] = None
    if "notes" not in df.columns:
        df["notes"] = ""
    
    return df


def save_timeline(csv_path: Path, df: pd.DataFrame) -> None:
    """
    Save the reviewed timeline with annotations back to disk.
    
    Args:
        csv_path (Path): Path where the CSV should be saved
        df (pd.DataFrame): DataFrame containing the timeline data
                         with review annotations
                         
    The function preserves all columns in the DataFrame, including
    the original emotion detection results and the added review
    annotations (goal opportunities and notes).
    """
    df.to_csv(csv_path, index=False)


def main():
    """
    Main application entry point for the Video Emotion Review Tool.
    
    This function implements the complete Streamlit web interface including:
    - Video selection sidebar
    - Filtering and navigation controls
    - Video playback with segment information
    - Review interface for goal opportunity annotation
    - Statistics display and data export functionality
    
    The interface is designed to be intuitive and efficient for reviewing
    large numbers of video segments, with features for tracking progress
    and maintaining consistency in annotations.
    """
    st.title("🎯 Video Emotion Review Tool")
    st.markdown("Review 'excited' segments to identify goal opportunities")
    
    # Sidebar for video selection
    timelines = get_available_timelines()
    
    if not timelines:
        st.error(f"No emotion timeline files found in '{OUTPUTS_DIR}' directory.")
        st.info("Run your emotion detection script first to generate timeline CSV files.")
        return
    
    video_names = sorted(timelines.keys())
    selected_video = st.sidebar.selectbox("Select Video", video_names)
    
    if selected_video:
        csv_path = timelines[selected_video]
        video_path = find_video_file(selected_video)
        
        if not video_path:
            st.error(f"Video file not found for '{selected_video}' in '{VIDEOS_DIR}' directory.")
            return
        
        # Load timeline
        df = load_timeline(csv_path)
        
        # Filter options
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filters")
        
        show_emotion = st.sidebar.multiselect(
            "Emotion Type",
            options=["excited", "neutral"],
            default=["excited"]
        )
        
        show_reviewed = st.sidebar.radio(
            "Review Status",
            options=["All", "Reviewed", "Not Reviewed"],
            index=0
        )
        
        # Apply filters
        filtered_df = df[df["emotion"].isin(show_emotion)].copy()
        
        if show_reviewed == "Reviewed":
            filtered_df = filtered_df[filtered_df["is_goal_opportunity"].notna()]
        elif show_reviewed == "Not Reviewed":
            filtered_df = filtered_df[filtered_df["is_goal_opportunity"].isna()]
        
        # Statistics
        st.sidebar.markdown("---")
        st.sidebar.subheader("Statistics")
        total_segments = len(df)
        excited_segments = len(df[df["emotion"] == "excited"])
        reviewed = len(df[df["is_goal_opportunity"].notna()])
        goal_opportunities = len(df[df["is_goal_opportunity"] == True])
        
        st.sidebar.metric("Total Segments", total_segments)
        st.sidebar.metric("Excited Segments", excited_segments)
        st.sidebar.metric("Reviewed", f"{reviewed}/{total_segments}")
        st.sidebar.metric("Goal Opportunities", goal_opportunities)
        
        # Main content
        st.header(f"📹 {selected_video}")
        
        if len(filtered_df) == 0:
            st.warning("No segments match the current filters.")
            return
        
        # Segment navigation
        st.markdown("---")
        segment_idx = st.number_input(
            f"Segment (1-{len(filtered_df)})",
            min_value=1,
            max_value=len(filtered_df),
            value=1,
            step=1
        ) - 1
        
        segment = filtered_df.iloc[segment_idx]
        original_idx = filtered_df.index[segment_idx]
        
        # Display segment info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Segment {segment_idx + 1} of {len(filtered_df)}")
            st.markdown(f"""
            - **Time**: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s
            - **Emotion**: {segment['emotion'].upper()}
            - **Intensity**: {segment['intensity']:.3f}
            """)
            
            # Video player using Streamlit's native video
            st.markdown("### Video Preview")
            st.info(f"⏱️ Segment starts at {segment['start_time']:.2f}s and ends at {segment['end_time']:.2f}s")
            st.video(str(video_path), start_time=int(segment['start_time']))
        
        with col2:
            st.subheader("Review")
            
            # Goal opportunity selection
            current_value = segment.get('is_goal_opportunity')
            if pd.isna(current_value):
                current_value = None
            
            is_goal = st.radio(
                "Is this a goal opportunity?",
                options=[True, False],
                format_func=lambda x: "✅ Yes" if x else "❌ No",
                index=0 if current_value is True else (1 if current_value is False else None),
                key=f"goal_{original_idx}"
            )
            
            # Notes
            current_notes = segment.get('notes', "")
            notes = st.text_area(
                "Notes (optional)",
                value=current_notes if pd.notna(current_notes) else "",
                key=f"notes_{original_idx}",
                height=100
            )
            
            # Save button
            if st.button("💾 Save Review", type="primary"):
                df.at[original_idx, 'is_goal_opportunity'] = is_goal
                df.at[original_idx, 'notes'] = notes
                save_timeline(csv_path, df)
                st.success("Review saved!")
                st.rerun()
            
            # Navigation buttons
            st.markdown("---")
            col_prev, col_next = st.columns(2)
            
            with col_prev:
                if segment_idx > 0:
                    if st.button("⬅️ Previous"):
                        st.session_state.segment_idx = segment_idx - 1
                        st.rerun()
            
            with col_next:
                if segment_idx < len(filtered_df) - 1:
                    if st.button("Next ➡️"):
                        st.session_state.segment_idx = segment_idx + 1
                        st.rerun()
        
        # Show all segments table
        st.markdown("---")
        st.subheader("All Filtered Segments")
        
        # Format display dataframe
        display_df = filtered_df.copy()
        display_df['is_goal_opportunity'] = display_df['is_goal_opportunity'].map(
            {True: "✅ Yes", False: "❌ No", None: "⏳ Pending"}
        )
        
        st.dataframe(
            display_df[['start_time', 'end_time', 'emotion', 'intensity', 'is_goal_opportunity', 'notes']],
            width='stretch'
        )
        
        # Export reviewed data
        st.markdown("---")
        if st.button("📥 Export Reviewed Data"):
            reviewed_df = df[df['is_goal_opportunity'].notna()].copy()
            csv = reviewed_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_video}_reviewed.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
