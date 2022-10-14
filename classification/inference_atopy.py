import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, datasets
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set gpu
print(device)
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def inference(opt,device):
    model_name = opt.model_name
    model = EfficientNet.from_pretrained(model_name, num_classes=opt.num_classes)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    model = model.to(device)
    fig = plt.figure()
    running_loss, running_corrects, num_cnt = 0.0, 0, 0

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    path=opt.img_path
    img=Image.open(path)
    # print(type(img))
    inputs=data_transforms(img)
    inputs = inputs.to(device)
    outputs = model(inputs.unsqueeze(0))
    sm = ()
    probabilities = torch.nn.functional.softmax(outputs,dim=1)
    top_p, top_class = probabilities.topk(5, dim=1)
    result = list(zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
    result.sort(key=lambda x: x[1])
    # print(result)
    class_names = {
                      "0": "atopic_dermatits",
                      "1": "seborrheic dermatitis",
                      "2": "psoriasis",
                      "3": "rosacea",
                      "4": "acne",
                  }
    dic = {}
    for i in result:
        class_prob, class_id=i
        class_name=class_names[str(class_id)]
        dic[class_name]=class_prob
    print(dic)
    return dic

def load_model(model_path):
    model_name = 'efficientnet-b7'
    model = EfficientNet.from_pretrained(model_name, num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    #
    model = model.to(device)
    return model
def classify_lesions(model,image):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img=Image.fromarray(image)
    inputs = data_transforms(img)
    inputs = inputs.to(device)
    outputs = model(inputs.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_p, top_class = probabilities.topk(5, dim=1)
    result = list(zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
    result.sort(key=lambda x: x[1])
    class_names = {
        "0": "atopic_dermatits",
        "1": "seborrheic dermatitis",
        "2": "psoriasis",
        "3": "rosacea",
        "4": "acne",
    }
    dic = {}
    for i in result:
        class_prob, class_id = i
        class_name = class_names[str(class_id)]
        dic[class_name] = class_prob
    # print(dic)
    return dic

if __name__ == '__main__':
    import skimage.draw
    model = load_model('./best_accuracy_88.85245901639345.pt')
    img_dir = '../test_images'
    dst_dir = '../segmentation/inference_result'
    for i in os.listdir(img_dir):
        print('*'*50)
        img_path = os.path.join(img_dir, i)
        print(img_path)
        image = skimage.io.imread(img_path)
        dic=classify_lesions(model,image)
        print('\tclassification results',dic)