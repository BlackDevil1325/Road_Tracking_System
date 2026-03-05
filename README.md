# Road Lane Detection System


## Abstract:
Road safety   is   always   an   area   that   concerned   many people   around   the   world   and   systems   that   aid   the drivers    have    been    appearing    ever    since    cars    and computers were combined  to  make  driving  safer  and more efficient.   There are plenty of systems that are able to warn  drivers  about  different  types  of  dangers:  lane departure,  collision  possibility  and various  traffic  signs. However,  there  is  still  room  for  development,  because modern  technologies,  like  the  rising  vision  about  the OPEN-CV,   allow   us   to   create   much   more   efficient systems.   Also,   the   detections   can   be   improved   to perform  better  in  various  situations,  such  as  different light  conditions,  road  quality,  etc.  In  this  project,  we present  the  plans  of  a  driver-assistance  system,  which will be capable of road lane  and traffic sign detection  by using an OPEN-CV.  

Lane coloration has become popular in real time vehicular ad-hoc networks (VANETs). The main emphasis of this paper is to find the further ways which can be used further to improve the result of lane detection algorithms. Noise, visibility etc. can reduce the performance or the existing lane detection algorithms. The methods developed so far are working efficiently and giving good results in case when noise is not present in the images. But problem is that they fail or not give efficient results when there is any kind of noise or fog in the road images. The noise can be anything like dust, shadows, puddles, oil stains, tire skid marks, etc.

It is developed through OpenCV (deep learning and classical CV) using Python.

## Features:
- Lane detection with averaged left and right lane lines
- Pothole detection using thresholding
- Support for images, videos, and live webcam feed
- Combined output showing both detections

## Getting Started

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate      # Windows
   # or: source venv/bin/activate    # Unix/macOS
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the detector** on one of the sample images:
   ```bash
   python lane_detection.py -i lane.jpeg -o result
   ```
   This will save a combined image with lane lines (green) and potholes (red rectangles).

4. **Run on video**:
   ```bash
   python lane_detection.py -i video.mp4 -o output.avi --video
   ```

5. **Run live from webcam**:
   ```bash
   python lane_detection.py -i webcam
   ```
   Press 'q' to quit.

## Enhancements:
- Improved lane detection by averaging lines to show distinct left and right lanes
- Video processing capabilities
- Real-time webcam support
- Combined visualization of multiple detections

You are welcome to extend the code with machine-learning models for better accuracy, traffic sign detection, or integration with vehicle systems.

![lane](https://user-images.githubusercontent.com/28294942/137758174-63d7c31d-b9f9-4c95-8295-559cf0ab2593.jpeg)