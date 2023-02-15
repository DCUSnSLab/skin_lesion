import sys
import logging
import cv2

import torch
import os
import numpy as np
import time
from setproctitle import *
sys.path.append('skin_segmentation')
sys.path.append('classification')
sys.path.append('segmentation')
from skin_segmentation import inference as skin_inference
from classification import inference_atopy as lesion_classification
from segmentation import lesion_segmentation

class skin_lesion:
    def __init__(self):
        self.cwd = os.getcwd()
        logging.basicConfig(level=logging.INFO)
        logging.info("Loading....")
        t1 = time.time()
        # self.skin_net = self.__load_skin_net()
        # self.cl_net = self.__load_cl_net()
        self.seg_net = self.__load_seg_net()
        logging.info(f'Loading Complete, Elapsed Time:  {time.time() - t1} ')

    def __load_skin_net(self):
        os.chdir('./skin_segmentation')
        skin_net = skin_inference.Skin_net()
        os.chdir(self.cwd)

        return skin_net
    def __load_cl_net(self):
        os.chdir('./classification')
        cl_net = lesion_classification.lesion_classification_net()
        os.chdir(self.cwd)
        return cl_net
    def __load_seg_net(self):
        os.chdir('./segmentation')
        seg_net = lesion_segmentation.lesion_segmentation_net()
        os.chdir(self.cwd)
        return seg_net

    def skin_segmentation(self, image_path=None, image=None,display=False):
        assert image_path or image, "skin inference에 이미지 또는 이미지의 경로를 넣어주세요"
        if (image_path != None):
            category, category_image, mask_image, boolean_mask = self.skin_net.inference(image_path=image_path)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # fixme: 아래 세 개의 주석은 원래 코드에 있던 이미지를 변형하는 코드입니다. 차후 성능이 낮다거나 어떤 문제가 생긴다면 해당 부분을 의심해야 합니다
            # image_width = (image.shape[1] // 32) * 32
            # image_height = (image.shape[0] // 32) * 32
            # image = image[:image_height, :image_width]
            category, category_image, mask_image, boolean_mask = self.skin_net.inference(image=image)
        if(display):
            cv2.imshow(category, category_image)
            # cv2.imshow(f'mask_{category}', mask_image)
            # cv2.imshow('dst', cv2.imread(image_path)* np.expand_dims(boolean_mask, axis=2))
            cv2.waitKey(3000)
        # todo: boolean_mask, 기본적으로 3채널 이미지라 생각하고 아래에 차원을 추가했습니다 grayscale일 경우는 차원 수를 추가하지 않아야합니다
        boolean_mask = np.expand_dims(boolean_mask, axis=2)
        # print(boolean_mask)
    def lesion_classification(self, image_path=None, image=None,images=None,display=False):
        if(image_path != None):
            self.cl_net.inference(image_path=image_path, display=display)
        elif(image != None):
            self.cl_net.inference(image=image, display=display)
        elif(images != None):
            for image in images:
                self.cl_net.inference(image=image, display=display)
    def lesion_segmentation(self, image_path=None, image=None,display=False,crop_scale = 1,save_path = None):
        return self.seg_net.inference(image_path=image_path, display=display, save_path=save_path,show_mask=True, show_bbox=True,
                                                 show_contour=False, show_label=False,crop_scale = crop_scale)
    def operation(self,image_path, display=False,crop_save_path = None, crop_scale = 1):
        '''
        음 이건 ?
        :param image_path:
        :param display:
        :param crop_save_path:
        :param crop_scale:
        :return:
        '''
        coordinates, roi_images = self.lesion_segmentation(image_path=image_path, display=display,crop_scale=3,save_path='./')
        self.lesion_classification()

        # print(coordinates)
        # print('*'*50)
        # print(roi_images)
        pass


if __name__ == '__main__':
    setproctitle('lesion')
    # image_path = './test_images/KakaoTalk_20221031_182210929.png'
    # image_path = './test_images/KakaoTalk_20221031_182210929.png'
    image_path = './test_images/220792306-0.png'
    skin_lesion = skin_lesion()
    skin_lesion.operation(image_path=image_path,display=True)
    # skin_lesion.skin_segmentation(image_path=image_path, display=True)
    # skin_lesion.lesion_segmentation(image_path=image_path, display=True,crop_scale=3,save_path='./')

    # skin_lesion.skin_inference(image_path='./test_images/NISI20211018_0000848901_web.jpg', display=True)
    # skin_lesion.classification(image_path='./test_images/238798900-2.png', display=False)
