import sys
import logging
import cv2

import torch
import os
import numpy as np
import time
sys.path.append('skin_segmentation')
sys.path.append('classification')
from skin_segmentation import inference as skin_inference
from classification import inference_atopy as lesion_classification

class skin_lesion:
    def __init__(self):
        self.cwd = os.getcwd()

        logging.basicConfig(level=logging.INFO)
        logging.info("Loading....")
        t1 = time.time()
        self.skin_net = self.__load_skin_net()
        logging.info(f'Loading Complete, Elapsed Time:  {time.time() - t1} ')

    def __load_skin_net(self):
        os.chdir('./skin_segmentation')
        skin_net = skin_inference.Skin_net()
        os.chdir(self.cwd)

        return skin_net

    def skin_inference(self, image_path=None, image=None,display=False):
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
            cv2.waitKey(6000)
        # todo: boolean_mask, 기본적으로 3채널 이미지라 생각하고 아래에 차원을 추가했습니다 grayscale일 경우는 차원 수를 추가하지 않아야합니다
        boolean_mask = np.expand_dims(boolean_mask, axis=2)



if __name__ == '__main__':
    skin_lesion = skin_lesion()
    skin_lesion.skin_inference(image_path='./test_images/shutterstock_1261397593.jpg',display=False)
