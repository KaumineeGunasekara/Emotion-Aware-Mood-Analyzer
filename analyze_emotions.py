import csv
from collections import Counter
import matplotlib.pyplot as plt

def analyze_and_plot(csv_file):
    times = []
    emotions = []
    confidences = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            times.append(int(row['Time (s)']))
            emotions.append(row['Emotion'])
            confidences.append(float(row['Confidence']))
    # Plot emotions over time
    plt.figure(figsize=(12, 4))
    plt.plot(times, emotions, marker='o', linestyle='-', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Emotion')
    plt.title('Emotion detected over time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Plot confidence over time
    plt.figure(figsize=(12, 3))
    plt.plot(times, confidences, marker='o', color='green', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Confidence (%)')
    plt.title('Emotion confidence over time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Summarize most frequent emotion
    count = Counter(emotions)
    most_common_emotion, freq = count.most_common(1)[0]
    print(f'Most frequent emotion: {most_common_emotion} ({freq} times)')
    print('Emotion frequencies:')
    for emotion, c in count.items():
        print(f'{emotion}: {c}')
