import streamlit as st
import pandas as pd
from emotion_engine import EmotionEngine
import cv2
import time

st.set_page_config(page_title="AI Mood Analyzer", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>🎭 AI Emotion & Mood Analyzer</h1>",
    unsafe_allow_html=True
)

engine = EmotionEngine()


if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

if "results_df" not in st.session_state:
    st.session_state.results_df = None



# ===============================
# Webcam Emotion Analysis
# ===============================

with st.expander("📷 Webcam Emotion Analysis", expanded=True):
    st.write("Analyze emotions live from your webcam (max 1 minute).")

    if "webcam_frames" not in st.session_state:
        st.session_state.webcam_frames = []

    if "webcam_start_time" not in st.session_state:
        st.session_state.webcam_start_time = None

    if "recording" not in st.session_state:
        st.session_state.recording = False

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Webcam"):
            st.session_state.webcam_frames = []
            st.session_state.webcam_start_time = time.time()
            st.session_state.recording = True

    with col2:
        if st.button("⏹ Stop Webcam"):
            st.session_state.recording = False

    frame_box = st.empty()

    # ---------- RECORDING LOOP ----------
    if st.session_state.recording:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_box.image(frame, channels="BGR")
            st.session_state.webcam_frames.append(frame)

            elapsed = time.time() - st.session_state.webcam_start_time

            # Auto-stop at 60 seconds
            if elapsed >= 60:
                st.session_state.recording = False
                break

            # Allow Streamlit to re-render buttons
            if not st.session_state.recording:
                break

        cap.release()
        frame_box.empty()

    # ---------- ANALYZE AFTER STOP ----------
    if (
        not st.session_state.recording
        and st.session_state.webcam_frames
    ):
        with st.spinner("Analyzing webcam data..."):
            df = engine.analyze_frames(
                st.session_state.webcam_frames
            )

            st.session_state.results_df = df
            engine.save_to_csv(df)

            st.success("Webcam analysis completed!")
            st.dataframe(df)

        # Clear frames so it doesn’t re-run
        st.session_state.webcam_frames = []

# Video Upload Analysis

st.header("🎥 Upload Video for Emotion Analysis")

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mpeg4"]
)

if uploaded_file is not None:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Analyze Uploaded Video"):
        with st.spinner("Analyzing video emotions..."):
            df = engine.analyze_video(video_path)

            st.session_state.results_df = df
            engine.save_to_csv(df)

            st.success("Video analysis completed!")
            st.dataframe(df)



# Emotion Summary Report

st.header("📊 Emotion Summary Report")

df = st.session_state.results_df

if df is not None and not df.empty:
    dominant_emotion = df["Emotion"].value_counts().idxmax()

    st.subheader(f"Overall Mood: **{dominant_emotion.upper()}**")
    st.bar_chart(df["Emotion"].value_counts())

    st.download_button(
        "⬇ Download CSV Report",
        data=df.to_csv(index=False),
        file_name="emotion_results.csv",
        mime="text/csv"
    )
else:
    st.info("Run webcam or video analysis to generate a report.")
