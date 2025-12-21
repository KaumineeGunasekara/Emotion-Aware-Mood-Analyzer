from emotion_engine import EmotionEngine
import analyze_emotions

engine = EmotionEngine(frame_interval=1)

df = engine.analyze_webcam(duration=300)
engine.save_to_csv(df)

analyze_emotions.analyze_and_plot("emotion_results.csv")
print("✅ Analysis complete!")
