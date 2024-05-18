# YOLO_NAS_Car_Logo_Detection
## About
This repository contains the files of my thesis titled "Enhancing Small Object Detection: Development on Car Logo Identification Using the YOLO-NAS Method for Speed and Accuracy Improvements". It includes the dataset used for training the model, the notebook, frontend, and backend files to run the web-based application. 

The web-based application consists of 3 pages:
- Home Page: Contains an introduction and news about YOLO-NAS and car logo detection
- Predict Page: Contains features for car logo detection using YOLO-NAS. Users can choose between 2 features,
- - Select from Folder: Users can choose images available in the folder each containing 10-13 images of challenges in car logo detection.
  - Input Image: Users can input an image manually from their local folder.
- Contact Page: Contains information on the author.

This is the link for the app's live demo:
https://huggingface.co/spaces/amadeomarvell/Car-Logo-Detection-YOLO-NAS

To run the app's front end, type 'streamlit run app.py'

To run the app's backend, type 'uvicorn backend_fastapi:app -- reload'
