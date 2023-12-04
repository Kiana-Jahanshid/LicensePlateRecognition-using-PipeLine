import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from deep_text_recognition_benchmark.utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition_benchmark.dataset import RawDataset, AlignCollate
from deep_text_recognition_benchmark.model import Model



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
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

opt = parser.parse_args()

if opt.sensitive:
    opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

class DTRB:
    def __init__(self , weights_path):

        if 'CTC' in opt.Prediction:
            self.converter = CTCLabelConverter(opt.character)
        else:
            self.converter = AttnLabelConverter(opt.character)
        opt.num_class = len(self.converter.character)

        if opt.rgb:
            opt.input_channel = 3     

        # select between cpu or gpu  , if gpu existed , use gpu 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(weights_path)
        


    def load_model(self , weights_path):
        self.model = Model(opt)
        print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
            opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
            opt.SequenceModeling, opt.Prediction)
        self.model = torch.nn.DataParallel(self.model).to(self.device)

        # load model
        print('loading pretrained model from %s' % weights_path)
        self.model.load_state_dict(torch.load(weights_path , map_location=self.device))


    def load_data(self , image_folder):
        # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        demo_data = RawDataset(root=image_folder, opt=opt)  # use RawDataset
        self.demo_loader = torch.utils.data.DataLoader(
            demo_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo, pin_memory=True)
    
    
    def predict(self , image):
        #self.load_data(image)
        # predict
        self.model.eval()
        with torch.no_grad():

            # normalizing with torch 
            transform = transforms.Compose([transforms.ToTensor() ,
                                            transforms.Normalize((0.5) , (0.5))])
            image_tensor = transform(image)


            # خط زیر برای تبدیل تصویر نامپای ایی به تنسور است :
            #image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
            print(image_tensor.shape) # printed as ---> torch.size()
            # printed output =  torch.Size([13, 81, 3])
            # we consume that first element of above array [13, 81, 3] , should be batch size ---> batch_size = image_tensor.size(0)
            # it means that it should be [ 1 , 13, 81, 3]
            # we should give 4D data to network .
            #for image_tensors, image_path_list in self.demo_loader:
            # <<<<<<<  batch size value is as same as the number of input images >>>>>>
            # so batch size is equal to 1 , bc we used only 1 image .

            # CONVERTING 3D ARRAY TO 4D ARRAY IS CALLED UNSQUEEZE
            # squeezing means = inserting one dimension to the tensor 
            # squeeze   : convert  3x3   ----> to  1x3x3
            # unsqueeze : convert  1x3x3 ----> to  3x3
            image_tensor = torch.unsqueeze(image_tensor , 0) # 0 means adding 1 into first index
            print("squeezed tensor (from 3D to 4D) : " , image_tensor.shape) # output is ----> torch.Size([1, 32 , 100])


            # resize : 
            # in dataset.py file we have this image shape  :     def __init__(self, imgH=32, imgW=100,
            # and these lines in train.py file :
            # parser.add_argument('--imgH', type=int, default=32
            # parser.add_argument('--imgW', type=int, default=100
            # یعنی شبکه یادگرفته پلاک هایی رو او سی آر بکنه که ابعادشون ۳۲در۱۰۰ باشن 
            # so we should resize image in main.py

            # shabake dare tasvir haye gray migire , va chon baad az gray kardan baz 1 dimension kam mishe 
            # pas bayad ye unsqueeze bezanim
            #image_tensor = torch.unsqueeze(image_tensor , 1)
            #print(image_tensor.shape)  # output  ---> torch.Size([1, 1, 32, 100])


            # normalizing 
            #image_tensor = image_tensor / 255.0



            batch_size = image_tensor.size(0)
            image = image_tensor.to(self.device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(self.device)

            if 'CTC' in opt.Prediction:
                preds = self.model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(preds_index, preds_size)

            else:
                preds = self.model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(["image"] , preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
        




if __name__ == '__main__':

    plate_recognizer = DTRB("../weights/DTRB-Recognizer/DTRB_best_accuracy_TPS_ResNet_BiLSTM_Attn.pth") # دیگه آدرس وزن هارو به عنوان آرگومان نمیدیم بلکه به عنوان پارامتر تابع سازنده میدیم 
    plate_recognizer.predict()



# 
#python DTRB_OO_v2.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder io/input_plates