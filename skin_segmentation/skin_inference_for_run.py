import argparse
import logging
import pathlib
import functools

import cv2
import torch
from torchvision import transforms


from .visualisation import draw_results

import torch
import torch.nn as nn
from .fcn import FCNResNet101
from .bisenetv2 import BiSeNetV2

models = {
    'FCNResNet101': FCNResNet101,
    'BiSeNetV2': BiSeNetV2,
}


def load_model(model_type, state_dict):
    # todo(will.brennan) - improve this... might want to save a categories file with this instead
    category_prefix = '_categories.'
    categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
    categories = [k[len(category_prefix):] for k in categories]

    model = model_type(categories)
    model.load_state_dict(state_dict)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    parser.add_argument('--model-type', type=str, choices=models, required=True)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f'*{file_ext}')


def _load_image(image_path: pathlib.Path):
    image = cv2.imread(str(image_path))
    assert image is not None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_width = (image.shape[1] // 32) * 32
    image_height = (image.shape[0] // 32) * 32

    image = image[:image_height, :image_width]
    return image


class Skin_net:
    def __init__(self,num=0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.device == 'cuda', "cuda가 안되는데요?, skin segmentation을 살펴보세요"

        __model = ['pretrained/model_segmentation_skin_30.pth', 'pretrained/model_segmentation_realtime_skin_30.pth']
        __model_tpye = ['FCNResNet101','BiSeNetV2']
        self.model = torch.load(__model[num], map_location=self.device)
        self.model = load_model(models[__model_tpye[num]], self.model)
        self.model.to(self.device).eval()
        self.image_transform = transforms.Compose(
            [
                transforms.Lambda(lambda image_path: _load_image(image_path)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.threshold = 0.5

    def inference(self, image_path):
        # image = fn_image_transform(image_path)
        image = self.image_transform(image_path)
        #
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)
            results = self.model(image)['out']
            results = torch.sigmoid(results)
            results = results > self.threshold
        for category, category_image, mask_image in draw_results(image[0], results[0], categories=self.model.categories):
            cv2.imshow(category, category_image)
            cv2.imshow(f'mask_{category}', mask_image)
            cv2.waitKey(0)



if __name__ == '__main__':

    skin_net = Skin_net(0)
    skin_net.inference(image_path='../test_images/2240583E56C5038C02.jpg')
