import cv2
from deepface import DeepFace
import time
import csv
import analyze_emotions
from advanced_layer import AdvancedAnalyzer 
import pandas as pd  

# Initialize
cap = cv2.VideoCapture(0)
duration = 300   # 5 minutes
start_time = time.time()
frame_interval = 1  # 1 second
emotion_results = []

# Initialize Advanced Analyzer
adv = AdvancedAnalyzer()

while int(time.time() - start_time) < duration:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = int(time.time() - start_time)

    if current_time % frame_interval == 0:
        try:
            # Emotion detection 
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion_scores = analysis[0]["emotion"]
            detected_emotion = analysis[0]["dominant_emotion"]
            confidence = emotion_scores[detected_emotion]
        except Exception as e:
            print("Error analyzing emotion:", e)
            detected_emotion = "Error"
            confidence = 0.0
            emotion_scores = {}

        # Advanced facial analysis 
        features = adv.analyze_frame(frame)
        if features is None:
            features = {
                "blink": False,
                "avg_eye_dist": None,
                "mouth_open": None,
                "mouth_width": None,
                "eyebrow_lift": None,
                "nose_x": None,
                "nose_flare": None
            }

        # Combine results into one row
        row = {
            "Time (s)": current_time,
            "Emotion": detected_emotion,
            "Confidence": confidence,
            "Blink": int(features["blink"]),
            "Eye_Distance": features["avg_eye_dist"],
            "Mouth_Open": features["mouth_open"],
            "Mouth_Width": features["mouth_width"],
            "Eyebrow_Lift": features["eyebrow_lift"],
            "Nose_X": features["nose_x"],
            "Nose_Flare": features["nose_flare"]
        }

        emotion_results.append(row)
        print(f"{current_time}s → {detected_emotion} ({confidence:.1f}%) | Blink:{row['Blink']} Mouth:{row['Mouth_Open']}")

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save all data to CSV
with open('emotion_results.csv', mode='w', newline='') as file:
    fieldnames = [
        "Time (s)", "Emotion", "Confidence",
        "Blink", "Eye_Distance", "Mouth_Open", "Mouth_Width",
        "Eyebrow_Lift", "Nose_X", "Nose_Flare"
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in emotion_results:
        writer.writerow(row)

#Plot results (same as before)
analyze_emotions.analyze_and_plot('emotion_results.csv')
print("✅ Advanced emotion analysis complete!")


# === FACIAL FEATURE SUMMARY ===
df = pd.read_csv('emotion_results.csv')

# Safely handle missing values
df = df.fillna(0)

total_time = df["Time (s)"].max() / 60 if len(df) > 0 else 1  # convert seconds to minutes

# Blink analysis
blink_count = df["Blink"].sum()
blink_rate = blink_count / total_time  # blinks per minute

# Smile detection (based on mouth_open threshold)
smile_count = (df["Mouth_Open"] > 0.03).sum()

# Look-away detection (nose_x too far left/right)
look_away_count = ((df["Nose_X"] < 0.35) | (df["Nose_X"] > 0.65)).sum()
look_away_percent = (look_away_count / len(df)) * 100 if len(df) > 0 else 0

# Eyebrow lift (smaller value means lifted)
eyebrow_raise_count = (df["Eyebrow_Lift"] < 0.02).sum()

# === Print Summary ===
print("\n--- Facial Feature Summary ---")
print(f"Total duration: {total_time:.1f} minutes")
print(f"Total blinks: {blink_count}  |  Blink rate: {blink_rate:.1f} blinks/min")
print(f"Smiles detected: {smile_count}")
print(f"Looked away: {look_away_percent:.1f}% of the time")
print(f"Eyebrow raises detected: {eyebrow_raise_count}")
print("-------------------------------")

# === MOOD IMPROVEMENT TIPS ===
print("\n💡 Mood Improvement Tips:")

# Find the most frequent emotion detected
dominant_mood = df["Emotion"].value_counts().idxmax()

print(f"\nYour overall mood seemed mostly: {dominant_mood.upper()}")

if dominant_mood == "happy":
    print("😊 Keep it up! Share your happiness with others or do something creative.")
    print("Tip: Maintain a gratitude journal to reinforce positive feelings.")
elif dominant_mood == "sad":
    print("😔 It seems you've felt a bit down.")
    print("Tip: Try listening to your favorite upbeat songs or go outside for fresh air.")
    print("Tip: Talking to a friend can also help lift your spirits.")
elif dominant_mood == "angry":
    print("😡 You seemed tense or frustrated.")
    print("Tip: Take a deep breath and step away from what's causing the anger.")
    print("Tip: Try a short walk or stretching to relax your body.")
elif dominant_mood == "fear":
    print("😨 There might be some anxiety or fear detected.")
    print("Tip: Try grounding yourself with slow, deep breathing.")
    print("Tip: Remind yourself that you are safe in this moment.")
elif dominant_mood == "surprise":
    print("😲 You seemed surprised or excited!")
    print("Tip: Channel that energy into learning or exploring something new.")
elif dominant_mood == "disgust":
    print("🤢 You may have experienced some discomfort or disapproval.")
    print("Tip: Reflect on what caused this feeling — is it something you can change or accept?")
else:
    print("😐 Your mood was mostly neutral.")
    print("Tip: Try a quick break — stretching, tea, or light music can refresh your mind.")
