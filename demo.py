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
        if (self.Config_yolo == None):
            raise Exception("please input yolo Configuration")
        self.yolo = self.__load_yolo()
        self.cl_41 = self.__load_cl_net(self.Config_41)
        self.cl_iga_grade = self.__load_cl_net(self.Config_iga_grade)
        self.cl_erythema = self.__load_cl_net(self.Config_erythema)
        self.cl_papulation = self.__load_cl_net(self.Config_papulation)
        self.cl_excoriation = self.__load_cl_net(self.Config_excoriation)
        self.cl_Lichenification = self.__load_cl_net(self.Config_lichenification)
        logging.info(f'Loading Complete, Elapsed Time:  {time.time() - t1} ')


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
        for i, data in enumerate(inference_results):
            cropped_image = data[1]

            # print(data[2])
            for index, cr in enumerate(data[1]):
                print("병변 분류", self.cl_41.inference(image=cr))
                print("중증도", self.cl_iga_grade.inference(image=cr))
                print("홍반", self.cl_erythema.inference(image=cr))
                print("구진", self.cl_papulation.inference(image=cr))
                print("줄까짐", self.cl_excoriation.inference(image=cr))
                print("태선화", self.cl_Lichenification.inference(image=cr))
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
        topk = 3
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
        model_path = "/home/dgdgksj/skin_lesion/classification/models/papulation.pt"
    class Config_excoriation(Config_erythema):
        model_path = "/home/dgdgksj/skin_lesion/classification/models/excoriation.pt"
    class Config_lichenification(Config_erythema):
        model_path = "/home/dgdgksj/skin_lesion/classification/models/lichenification.pt"
    skin_lesion = Skin_lesion(Config_yolo=Config_yolo,Config_41=Config_41,Config_iga_grade=Config_iga_grade,Config_erythema=Config_erythema,Config_papulation=Config_papulation,Config_excoriation=Config_excoriation,Config_lichenification=Config_lichenification)
    skin_lesion.inference(image_path="/home/dgdgksj/skin_lesion/ultralytics/test_images/KakaoTalk_20221031_182210929.png")
    # # image_path = './test_images/KakaoTalk_20221031_182210929.png'
    # # image_path = './test_images/KakaoTalk_20221031_182210929.png'
    # image_path = './test_images/220792306-0.png'
    # skin_lesion = skin_lesion()
    # skin_lesion.operation(image_path=image_path,display=True)
