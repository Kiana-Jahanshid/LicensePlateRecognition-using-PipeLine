import cv2
from ultralytics import YOLO
from deep_text_recognition_benchmark.DTRB_OO_v2 import DTRB


image = cv2.imread("io/input/13.jpg")
#DETECTION module
plate_detector = YOLO("weights/YOLOv8-Detector/YOLOv8n_license_plate_detector_best_weight.pt")
#RECOGNITION module
plate_recognizer = DTRB("weights/DTRB-Recognizer/DTRB_best_accuracy_TPS_ResNet_BiLSTM_Attn.pth") # دیگه آدرس وزن هارو به عنوان آرگومان نمیدیم بلکه به عنوان پارامتر تابع سازنده میدیم 

result = plate_detector.predict(image)
result = result[0]

# draw a green rectangle around license plate
for i in range(len(result.boxes.xyxy)) :
    if result.boxes.conf[i] > 0.8 :
        bounding_box = result.boxes.xyxy[i]
        print(bounding_box)
        # first we have to convert TENSOR to numpy array and then use it :
        bounding_box = bounding_box.cpu().detach().numpy().astype(int) # now bounding_box is a numpy array
        print("numpy array : " , bounding_box)
        cropped_plate_img = image[bounding_box[1]:bounding_box[3]  ,  bounding_box[0]:bounding_box[2]].copy()
        cropped_plate_img = cv2.resize(cropped_plate_img , (100,32)) ## refer to line 120 in DTRB_OO_v2.py
        cv2.imwrite(f"io/output/x_cropped{i}.jpg" , cropped_plate_img)
        cropped_plate_img = cv2.cvtColor(cropped_plate_img , cv2.COLOR_BGR2GRAY)
        cv2.rectangle(image , (bounding_box[0],bounding_box[1]) , (bounding_box[2] , bounding_box[3]) , (0,255,0) , 4)
        plate_recognizer.predict(cropped_plate_img)

cv2.imwrite("io/output/x.jpg" , image)
 
 
'''''
if we had segmentation problem : masks = value - except None
if we had pose detection problem : keypoints would have value or else it gets None
now we have object detection , therefore boxes variable gets value : ultralytics.engine.results.Boxes object

چون داریم فقط روی یک تصویر پلاک ریکاگنیشتن انجام میدیم 
میریم توی فایل دی تی آر بی شی گرا شده و در متد پردیکت اسم پارامتر ایمج فولدر رو 
به ایمج خالی تغیر میدیم .
و متد لود دیتا رو هم کال نمی کنیم چون دیگه برای فقط یک تصویر نیاز به شافل و بچ سایز و ... نداریم 
اسم ایمج تنسورز رو هم به ایمج تنسور تغییر دادیم 

ما در فایل مین پلاک کراپ شده رو داریم به متد پردیکت میدیم که از جنس نامپای ارری هست 
اما توی متد پردیکت تصویر رو به شکل تنسور از ما میخواد 
پس باید تصویر نامپای شده رو دوباره برگردونیم به تنسور !!!!

'''''


#python main.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn