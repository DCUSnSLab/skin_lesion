from ultralytics import YOLO
import numpy as np
import cv2

from setproctitle import *
import os
from copy import deepcopy

class Lesion_segmentation_with_yolo:
    def __init__(self,config):
        self.model = YOLO(config.model_path)
        self.Config = config
        self.Config.model_names = self.model.names

    def __get_image_paths(self, image_path):
        if (os.path.isfile(image_path)):
            self.Config.file_paths = [image_path]
        elif (os.path.isdir(image_path)):
            file_paths = [x for x in os.listdir(image_path)]
            for i, file in enumerate(file_paths):
                if ('.png' in file or '.jpg' in file):
                    pass
                else:
                    file_paths[i] = None
            temp_indexes = list(np.where(np.array(file_paths) != None)[0])
            file_paths = [os.path.join(image_path, file_paths[x]) for x in temp_indexes]
            if (len(file_paths) > 0):
                self.Config.file_paths = file_paths
            else:
                raise Exception("No valid image files found, please check dir")
        else:
            raise Exception("image_path is not dir or valid image, please check image_path")
    def __draw_boexes(self,image, boxes):
        cropped_images = [None for _ in range(len(boxes.boxes))]
        for i, box in enumerate(boxes.boxes):
            lw = max(round(sum(image.shape) / 2 * 0.003), 2)
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cropped_images[i] = self.copied_image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            cv2.rectangle(image, p1, p2, (0, 0, 255), thickness=lw, lineType=cv2.LINE_AA)
            cur_class = self.Config.model_names[int(boxes.cls[i])]

            if self.Config.label:
                cur_label = '' + cur_class
                tf = max(lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(cur_label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(image, p1, p2, (0, 0, 255), -1, cv2.LINE_AA)  # filled
                cv2.putText(image, cur_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3,
                            (255, 255, 255),
                            thickness=tf, lineType=cv2.LINE_AA)
        return image, cropped_images

    def __draw_segmentation(self, image, segmentations, alpha=0.8):
        for segmentation in segmentations:
            segmentation = segmentation.cpu().numpy()
            segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
            color = np.array([0, 0, 255], dtype='uint8')
            masked_img = np.where(segmentation[..., None], color, image)
            image = cv2.addWeighted(src1=image, alpha=alpha, src2=masked_img, beta=0.2, gamma=0)
        # if (True):
        #     cv2.imshow("image", image)
        #     cv2.waitKey(30000)
        return image

    def __get_results(self, cur_image_path):
        image = cv2.imread(cur_image_path)
        self.copied_image = deepcopy(image)
        results = self.model.predict(cur_image_path, device=self.Config.device, verbose=self.Config.verbose)
        length = len(results[0].boxes.cpu().numpy())
        cropped_images = image
        if (length <= 0):
            print('\033[31m' + "object is not detected!!!", "image_path: " + cur_image_path + '\033[0m')
            return image, [cropped_images], False
        # raise Exception('\033[31m' + "object is not detected!!!", "image_path: " + image_path + '\033[0m')
        else:
            for i, result in enumerate(results):
                image,cropped = self.__draw_boexes(image=image, boxes=result.boxes)
                cropped_images = cropped
                image = self.__draw_segmentation(image, segmentations=result.masks.masks)
        return image, cropped_images, True


    def inference(self, image_path,save_path=False):
        self.__get_image_paths(image_path)

        inference_results = [None for _ in range(len(self.Config.file_paths))]

        for i, image_path in enumerate(self.Config.file_paths):
            image, cropped_images,status = self.__get_results(cur_image_path=image_path)
            inference_results[i] = (image,cropped_images,status)

            if (save_path):
                raise NotImplementedError("save function is not implemented ㅜㅜ")
            if (self.Config.display):
                cv2.imshow("image", image)
                cv2.waitKey(1000)
        return inference_results





if __name__ == '__main__':
    setproctitle('yolo test')
    # model_path = "/home/dgdgksj/skin_lesion/ultralytics/best_n.pt"
    # # model_path = "yolov8n-seg.pt"
    # # image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/KakaoTalk_20221031_182210929.png"
    # # image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/2240583E56C5038C02.jpg"
    # # image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/bus.jpg"
    # image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/"
    # lesion_yolo = Lesion_segmentation_with_yolo(model_path=model_path)
    # inference_results = lesion_yolo.inference(image_path=image_path, display=False)
    # for i, data in enumerate(inference_results):
    #     for index,cr in enumerate(data[1]):
    #         cv2.imshow(str(index),cr)
    #
    #     cv2.imshow("results",data[0])
    #     # print(len(data[1]))
    #     # cv2.imshow("results2", img[1][0)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    # # image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/KakaoTalk_20221031_182210929.png"
    #
    # pass

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

    image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/KakaoTalk_20221031_182210929.png"
    image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/"
    lesion_yolo = Lesion_segmentation_with_yolo(config=Config_yolo)
    inference_results = lesion_yolo.inference(image_path=image_path)
    for i, data in enumerate(inference_results):
        for index,cr in enumerate(data[1]):
            cv2.imshow(str(index),cr)

        cv2.imshow("results",data[0])
        # print(len(data[1]))
        # cv2.imshow("results2", img[1][0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()