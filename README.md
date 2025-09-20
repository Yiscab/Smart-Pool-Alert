# AI Pool Guard Demo

This repository contains a demo for a smart video-based pool safety system. The goal is to detect and alert on unsafe behavior of children near or inside a swimming pool using computer vision and basic simulation techniques.

## 📊 Project Overview

This demo simulates two possible safety scenarios:

1. **Walking Near the Pool (Supervision Alert)**:
   - A child walks around the pool unsupervised.
   - The system detects the presence and sends an alert before an accident can occur.

2. **In-Water Distress (Drowning Alert)**:
   - A doll (representing a child) moves irregularly and then stops, simulating drowning.
   - The system analyzes motion patterns and raises an alert if distress is detected.

## 🔧 Features

- Simulation of video scenarios using `OpenCV`.
- Region of Interest (ROI) editor for focusing on the pool area.
- Alert sound generator using `playsound`.
- Custom detection logic for movement and inactivity.
- Ready for future integration with real camera feeds or AI models.

## 🌂 Folder Structure

```
project-folder/
├── generate_distress_video.py     # Script to generate distress video
├── edit_roi_interactive.py       # Tool for editing the ROI polygon
├── detect_person_roi.py          # Main alert logic over video input
├── doll.png / doll_2.png         # Doll sprite with transparency
├── roi_doll.json                # Saved polygon coordinates
├── alert.wav                    # Audio alert to play
└── doll_distress.mp4             # Generated video of the pool scenario
```

## 🔢 How to Run

1. **Create Video**:

```bash
python generate_distress_video.py
```

2. **Define Pool Region (Optional)**:

```bash
python edit_roi_interactive.py
```

3. **Run Main Detection System**:

```bash
python detect_person_roi.py
```

You should see the video running and alerts triggered when needed.

## ⚠️ Requirements

- Python 3.8+
- OpenCV
- NumPy
- playsound

Install dependencies:

```bash
pip install -r requirements.txt
```

(You can create this file manually with required libraries.)

## 🔍 Next Steps

- [ ] Add second scenario (walking near pool)
- [ ] Improve detection logic using AI (optional)
- [ ] Package into Streamlit app / GUI
- [ ] Upload to GitHub with proper README and license

## 🌟 Credits

Created by Yisca Biton for an educational AI safety project.
All images used are AI-generated or licensed for demo purposes.

---

🚀 Ready for integration with edge devices or AI models in the future.

