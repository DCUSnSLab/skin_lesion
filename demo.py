import sys
import logging
import cv2

import torch
import os
import numpy as np
import time
from setproctitle import *

sys.path.append('classification')
from classification import inference_atopy as lesion_classification
from run_class_based_yolo import Lesion_segmentation_with_yolo



from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
class Efficientnet_Ensemble(nn.Module):
    def __init__(self, model_iga_grade, model_erythema, model_papulation, model_excoriation, model_Lichenification):
        super(Efficientnet_Ensemble, self).__init__()
        self.model_iga_grade = model_iga_grade.model
        self.model_erythema = model_erythema.model
        self.model_papulation = model_papulation.model
        self.model_excoriation = model_excoriation.model
        self.model_Lichenification = model_Lichenification.model

    def forward(self, x):
        # t1= time.time()
        # x1 = self.model_iga_grade(x.clone())
        # x2 = self.model_erythema(x.clone())
        # x3 = self.model_papulation(x.clone())
        # x4 = self.model_excoriation(x.clone())
        # x5 = self.model_Lichenification(x.clone())

        x1 = F.softmax(self.model_iga_grade(x.clone()), dim=1).topk(4, dim=1)  # clone to make sure x is not changed by inplace methods
        x2 = F.softmax(self.model_erythema(x.clone()), dim=1).topk(4, dim=1)
        x3 = F.softmax(self.model_papulation(x.clone()), dim=1).topk(4, dim=1)
        x4 = F.softmax(self.model_excoriation(x.clone()), dim=1).topk(4, dim=1)
        x5 = F.softmax(self.model_Lichenification(x.clone()), dim=1).topk(4, dim=1)
        # t2 = time.time() - t2
        # t3 = time.time()
        # x1 = list(
        #     zip([t.item() for t in x1[0].squeeze().squeeze()], [t.item() for t in x1[1].squeeze().squeeze()]))
        # x2 = list(
        #     zip([t.item() for t in x2[0].squeeze().squeeze()], [t.item() for t in x2[1].squeeze().squeeze()]))
        # x3 = list(
        #     zip([t.item() for t in x3[0].squeeze().squeeze()], [t.item() for t in x3[1].squeeze().squeeze()]))
        # x4 = list(
        #     zip([t.item() for t in x4[0].squeeze().squeeze()], [t.item() for t in x4[1].squeeze().squeeze()]))
        # x5 = list(
        #     zip([t.item() for t in x5[0].squeeze().squeeze()], [t.item() for t in x5[1].squeeze().squeeze()]))
        # t3 = time.time() - t3
        # print(t1,t2,t3)
        # print(time.time()-t1)
        # x = torch.cat((x1, x2), dim=1)

        # x = torch.cat((x1, x2), dim=1)
        # x = self.classifier(F.relu(x))
        return x1, x2, x3, x4, x5



class Skin_lesion:
    def __init__(self,Config_yolo=None,Config_41=None,Config_iga_grade=None,Config_erythema=None,Config_papulation=None,Config_excoriation=None,Config_lichenification=None):
        self.cwd = os.getcwd()
        self.Config_yolo = Config_yolo
        self.Config_41 = Config_41
        self.Config_iga_grade = Config_iga_grade
        self.Config_erythema = Config_erythema
        self.Config_papulation = Config_papulation
        self.Config_excoriation = Config_excoriation
        self.Config_lichenification = Config_lichenification
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading....")
        t1 = time.time()
        self.__check_config()
        self.yolo = self.__load_yolo()
        self.cl_41 = self.__load_cl_net(self.Config_41)
        self.cl_iga_grade = self.__load_cl_net(self.Config_iga_grade)
        self.cl_erythema = self.__load_cl_net(self.Config_erythema)
        self.cl_papulation = self.__load_cl_net(self.Config_papulation)
        self.cl_excoriation = self.__load_cl_net(self.Config_excoriation)
        self.cl_Lichenification = self.__load_cl_net(self.Config_lichenification)
        self.ensemble = Efficientnet_Ensemble(self.cl_iga_grade,self.cl_erythema,self.cl_papulation,self.cl_excoriation,self.cl_Lichenification)
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logging.info(f'Loading Complete, Elapsed Time:  {time.time() - t1} ')

    def __check_config(self):
        if (self.Config_yolo == None):
            raise Exception("please check yolo Configurations")
        elif (self.Config_41 == None):
            raise Exception("please check 41 Configurations")
        elif (self.Config_iga_grade == None):
            raise Exception("please check iga_grade Configurations")
        elif (self.Config_erythema == None):
            raise Exception("please check erythema Configurations")
        elif (self.Config_papulation == None):
            raise Exception("please check papulation Configurations")
        elif (self.Config_excoriation == None):
            raise Exception("please check excoriation Configurations")
        elif (self.Config_lichenification == None):
            raise Exception("please check lichenification Configurations")

    def __load_cl_net(self,Config):
        os.chdir('./classification')
        cl_net = lesion_classification.classification_net(Config=Config)
        os.chdir(self.cwd)
        return cl_net
    def __load_yolo(self):
        return Lesion_segmentation_with_yolo(self.Config_yolo)

    def inference(self, image_path):

        inference_results = self.yolo.inference(image_path=image_path)

        # print(len(inference_results))
        # print()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i, data in enumerate(inference_results):
            cropped_image = data[1]

            # print(data[2])
            for index, cr in enumerate(data[1]):
                t1=time.time()
                self.cl_41.inference(image=cr)
                self.cl_iga_grade.inference(image=cr)
                self.cl_erythema.inference(image=cr)
                self.cl_papulation.inference(image=cr)
                self.cl_excoriation.inference(image=cr)
                t1=time.time()-t1
                # print("병변 분류", self.cl_41.inference(image=cr))
                # print("중증도", self.cl_iga_grade.inference(image=cr))
                # print("홍반", self.cl_erythema.inference(image=cr))
                # print("구진", self.cl_papulation.inference(image=cr))
                # print("줄까짐", self.cl_excoriation.inference(image=cr))
                # print("태선화", self.cl_Lichenification.inference(image=cr))

                t2 = time.time()
                cr= cv2.cvtColor(cr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(cr).convert('RGB')

                inputs = self.data_transforms(image)
                inputs = inputs.to(device)
                outputs = self.ensemble(inputs.unsqueeze(0))
                t2 = time.time()-t2
                print(t1,t2)
                # probabilities = torch.nn.functional.softmax(outputs, dim=1)
                for output in outputs:
                    # print(output)
                    # print(torch.nn.functional.softmax(output, dim=1))
                    # print('*'*50)
                    pass
                # print(probabilities)
                # cv2.imshow(str(index), cr)
            #
            # cv2.imshow("results", data[0])
            # # print(len(data[1]))
            # # cv2.imshow("results2", img[1][0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        pass


if __name__ == '__main__':
    setproctitle('lesion')

    class Config_yolo():
        model_path = "/home/dgdgksj/skin_lesion/ultralytics/best_n.pt"
        model_names = None
        display = False
        save_path = None
        verbose = False
        device = 0
        label = True
        bbox = True
        segmentation = True
        file_paths = None
    class Config_41():
        verbose = False
        topk = 3
        model_path = "/home/dgdgksj/skin_lesion/classification/models/41.pt"
        model_name = 'efficientnet-b0'
        class_names = {
        "000": "normal_skin",
        "001": "atopy",
        "002": "prurigo",
        "003": "scar",
        "004": "psoriasis",
        "005": "varicella",
        "006": "nummular_eczema",
        "007": "ota_like_melanosis",
        "008": "becker_nevus",
        "009": "pyogenic_granuloma",
        "010": "acne",
        "011": "salmon_patches",
        "012": "dermatophytosis",
        "013": "wart",
        "014": "impetigo",
        "015": "vitiligo",
        "016": "ingrowing_nails",
        "017": "congenital_melanocytic_nevus",
        "018": "keloid",
        "019": "epidermal_cyst",
        "020": "insect_bite",
        "021": "molluscum_contagiosum",
        "022": "pityriasis_versicolor",
        "023": "melanonychia",
        "024": "alopecia_areata",
        "025": "epidermal_nevus",
        "026": "herpes_simplex",
        "027": "urticaria",
        "028": "nevus_depigmentosus",
        "029": "lichen_striatus",
        "030": "mongolian_spot_and_ectopic_mongolian_spot",
        "031": "capillary_malformation",
        "032": "pityriasis_lichenoides_chronica",
        "033": "infantile_hemangioma",
        "034": "mastocytoma",
        "035": "nevus_sebaceous",
        "036": "onychomycosis",
        "037": "milk_coffee_nevus",
        "038": "nail_dystrophy",
        "039": "melanocytic_nevus",
        "040": "juvenile_xanthogranuloma",
        }
    class Config_iga_grade():
        # 중증도
        verbose = False
        topk = 4
        model_path = "/home/dgdgksj/skin_lesion/classification/models/iga_grade.pt"
        model_name = 'efficientnet-b0'
        class_names = {
            "000": "Almost_Clear",
            "001": "Mild",
            "002": "Moderate",
            "003": "Severe",
        }
    class Config_erythema():
        # 홍반
        verbose = False
        topk = 1
        model_path = "/home/dgdgksj/skin_lesion/classification/models/erythema.pt"
        model_name = 'efficientnet-b0'
        class_names = {
            "000": "None",
            "001": "Mild",
            "002": "Moderate",
            "003": "Severe",
        }
    class Config_papulation(Config_erythema):
        # 구진
        model_path = "/home/dgdgksj/skin_lesion/classification/models/papulation.pt"
    class Config_excoriation(Config_erythema):
        # 줄까짐
        model_path = "/home/dgdgksj/skin_lesion/classification/models/excoriation.pt"
    class Config_lichenification(Config_erythema):
        # 태선화
        model_path = "/home/dgdgksj/skin_lesion/classification/models/lichenification.pt"
    skin_lesion = Skin_lesion(Config_yolo=Config_yolo,Config_41=Config_41,Config_iga_grade=Config_iga_grade,Config_erythema=Config_erythema,Config_papulation=Config_papulation,Config_excoriation=Config_excoriation,Config_lichenification=Config_lichenification)
    skin_lesion.inference(image_path="/home/dgdgksj/skin_lesion/ultralytics/test_images/KakaoTalk_20221031_182210929.png")
    # # image_path = './test_images/KakaoTalk_20221031_182210929.png'
    # # image_path = './test_images/KakaoTalk_20221031_182210929.png'
    # image_path = './test_images/220792306-0.png'
    # skin_lesion = skin_lesion()
    # skin_lesion.operation(image_path=image_path,display=True)
