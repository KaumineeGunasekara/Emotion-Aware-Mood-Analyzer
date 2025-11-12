# 🎭 Emotion-Aware Mood Analyzer

An AI-based system that detects facial emotions and expressions using **DeepFace** and **Mediapipe FaceMesh**.  
It analyzes a user's **emotions**, **eye blinks**, **mouth movements**, **eyebrow lifts**, and **gaze direction** from a webcam or video feed — then visualizes emotion trends and gives **personalized mood improvement tips**.

---

## 🚀 Features
- 🎥 **Real-time emotion detection** using DeepFace (happy, sad, angry, neutral, fear, surprise, disgust)
- 🧠 **Facial feature analysis** (blink rate, smile detection, eyebrow lift, gaze direction)
- 📊 **Emotion trend visualization** using Matplotlib
- 🗂️ **Automatic data storage** in CSV for each session
- 💡 **Mood improvement suggestions** based on your dominant emotion
- ⏱️ Designed for **5-minute analysis sessions**

---

## 🧩 Tech Stack
- **Python 3.11**
- **OpenCV** – video capture
- **DeepFace** – emotion recognition
- **Mediapipe** – facial landmarks & feature tracking
- **Matplotlib / Pandas** – visualization & data analysis

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone //https://github.com/KaumineeGunasekara/Emotion-Aware-Mood-Analyzer.git
cd Emotion-Aware-Mood-Analyzer
```
### 2️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
###3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
###4️⃣ Run the project
```bash
python main.py
```

Press 'q' to exit the webcam early.

---
## 📊 Output Files
- **emotion_results.csv** → detailed frame-by-frame data  
- **dominant_emotion_timeline.png** → timeline chart of detected emotions  
- **mood_tips.txt** → generated mood improvement advice  

---

## 👩‍💻 Author
**Kauminee Gunasekara**
