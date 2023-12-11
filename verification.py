import cv2
import glob
import string
import torch
import sqlite3
import argparse
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from difflib import SequenceMatcher
from main2 import verify_license_plate
from deep_text_recognition_benchmark.DTRB_OO_v2 import DTRB

parser = argparse.ArgumentParser()
#parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
#parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
""" our argument parsers """
parser.add_argument('--detector_wights' ,type=str , default="weights/YOLOv8-Detector/YOLOv8n_license_plate_detector_best_weight.pt")
parser.add_argument('--recognizer_weights' , type=str ,default="weights/DTRB-Recognizer/DTRB_best_accuracy_TPS_ResNet_BiLSTM_Attn.pth" )
parser.add_argument('--input_images' , type=str , default="io/input/*.jpg")
parser.add_argument('--threshold' , type=float , default=0.7)

opt = parser.parse_args()
if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()


con = sqlite3.connect("license_plates.db")

# read data from database 
input_image = cv2.imread(opt.input_images)
test_image_text = verify_license_plate(input_image )
df = pd.read_sql_query("SELECT * from license_plates", con)
df = pd.DataFrame(df)
print("license plate text  : " , test_image_text)


# compare our test license plate image , with data in DB 
flag = False
for i in range(len(df.text)) :
    if SequenceMatcher(None,df.text[i] , test_image_text).ratio() > 0.9 :
        print(df[df['text'] == test_image_text])
        flag = True 
if flag == True: 
    print("TRUE 🟢🟢🟢 this license plate existed in database " )
else :
    print("FALSE 🔴🔴🔴 this license plate didn't exist in database " )


con.close()



#python verification.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --input_images io/test2.jpg
