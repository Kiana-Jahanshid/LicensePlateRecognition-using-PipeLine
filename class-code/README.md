# OCR with pipeline and Object Oriented DTRB:


# Description :
![](assets/capture.JPG)

# How to install :
```
pip install -r requirements.txt
```
# How to run :
```
python main.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn 
```
or 
```
python deep_text_recognition_benchmark/DTRB_OO.py 
--Transformation TPS --FeatureExtraction ResNet 
--SequenceModeling BiLSTM --Prediction Attn 
--image_folder io/input_plates 
--saved_model weights/DTRB-Recognizer/DTRB_best_accuracy_TPS_ResNet_BiLSTM_Attn.pth 
```

# Results :

+ trained weights :
YOLO detector : <br/>
```
https://drive.google.com/file/d/1ki7GNd_3zJ8bUIEBBc8tkyiVvMEJZsvs/view?usp=drive_link

```
DTRB recognizer : <br/>
```
https://drive.google.com/file/d/1CI6C9ButxSbk8FdWLCwl-70z63sRLci6/view?usp=drive_link
```



<br/>

|                Ground Truth                 | predicted_labels |
|:-------------------------------------:| :-------------------------------------:| 
| ![](io/input_plates/1.jpg "1") | 63m45154 |
| ![](io/input_plates/2.jpg "1") | 33t83622 |
| ![](io/input_plates/3.jpg "1") | 11a1211 |
| ![](io/input_plates/4.jpg "1") | 56e88644 |
| ![](io/input_plates/5.jpg "1") | 44b12135 |
| ![](io/input_plates/8.jpg "1") | 88e32768 | 
| ![](io/input_plates/6.jpg "1") | 21a43513 | 