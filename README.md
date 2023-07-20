# MAGIC Mediapipe Pose Analyser Application for Lunges

### Project Overview
This project counts the number of lunges for performed on each leg. It also provides some advice to the trainee regarding form, although improvements need to be made to this.
To count the number of lunges performed two heuristics are used, these are the height of the hip compared to the height of the knee for the front leg as well as the angle of the knee for the front leg. 
Each heuristic works better depending on the angle of the user relative to the camera.

### Key File
`./mediapipe/examples/pose_landmarker/android/app/src/main/java/com/google/mediapipe/examples/poselandmarker/OverlayView.kt`
This file includes the computer vision code for the project allowing lunge reps to be counted.