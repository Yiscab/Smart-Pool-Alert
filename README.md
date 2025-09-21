# Smart Pool Safety System – Real-Time Drowning Alert 🏊‍♀️🚨

An AI-powered safety system designed to **prevent drowning incidents** near swimming pools.  
The system detects when a child approaches or remains in a restricted pool area, and also aims to **identify early signs of drowning** based on motion and behavior analysis.

---

## Project Goals
- Prevent child access to pool areas without supervision
- Detect abnormal behavior such as **lack of movement**, **lingering in the pool**, or **distress**
- Trigger immediate alerts to enable rapid response
- Serve as an accessible and affordable software-based safety solution

---

## Features
- YOLO-based object detection to identify people near the pool
- Real-time video feed processing
- Configurable **Region of Interest (ROI)** for danger zone
- Audio alert (e.g. siren) when a person enters the ROI
- Motion behavior analysis to detect possible drowning signs (planned/ongoing)
- Easy integration and customization

---

## How it Works

Define the pool ROI (Region of Interest):

Option 1 – Auto detection:
Using auto_water_detector.py, the system automatically detects water regions (e.g. pool) based on color segmentation and saves them into pool_roi.json.

Option 2 – Manual selection (optional):
Using edit_roi_interactive.py, you can manually draw the ROI on the camera feed and save it as roi_child.json.

Run main detection script:
Start live video monitoring with poolwatch_main.py, which loads the defined ROI from the JSON file.

The system continuously checks for:

Person entering the ROI → triggers alert

Lack of movement or abnormal patterns → flagged for future drowning detection logic

Plays alert sound via distress_alert.wav

---

## Tech Stack
- Python 3.x
- OpenCV
- Numpy
- YOLO (weights/model not included)
- JSON for storing ROI configuration
- Simpleaudio for audio playback

---

## Project Structure

```
project-folder/
├── poolwatch_main.py              # Script to generate distress video
├── detect_person_roi.py            # Person detection logic
├── edit_roi_interactive.py        # Tool to define ROI
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── alert.wav                    # Audio alert to play
└── demo/record_poolWatch1.mp4       # Live screen recording demo          

```
##  Demo Video
A short screen recording of the system running live, detecting a child near the pool and triggering alerts:
[Click here to watch the demo](./demo/record_poolWatch1.mp4)


---

## Disclaimer
Some core logic and the full drowning behavior model are intentionally omitted due to potential commercial development.  
Full demo and implementation details available upon request.

---

## Contact
For questions, collaboration or demo access:  
[yb0533144497@gmail.com]
[LinkedIn Profile](https://www.linkedin.com/in/yisca-biton-638932228/)

---

## Created By
Yisca Biton – Embedded & System Engineer  
This project was fully designed and developed as part of a personal initiative to explore AI-based safety systems.  

© 2025 Yisca Biton. All rights reserved.