from ultralytics import YOLO
from deep_text_recognition_benchmark.DTRB_OO_v2 import DTRB

#DETECTION
plate_detector = YOLO("weights/YOLOv8-Detector/YOLOv8n_license_plate_detector_best_weight.pt")
plate_detector.predict("io/input/" , save=True , save_crop=True)

#RECOGNITION
plate_recognizer = DTRB("weights/DTRB-Recognizer/DTRB_best_accuracy_TPS_ResNet_BiLSTM_Attn.pth") # دیگه آدرس وزن هارو به عنوان آرگومان نمیدیم بلکه به عنوان پارامتر تابع سازنده میدیم 
plate_recognizer.predict("io/input_plates")