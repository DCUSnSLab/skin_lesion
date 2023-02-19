import numpy as np
import torch.nn as nn
import skimage.draw
import matplotlib.pyplot as plt
import os
from efficientnet_pytorch import EfficientNet
from torchvision import transforms, datasets
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.onnx
from torchvision import models
import onnx
from onnx import shape_inference
import onnx.numpy_helper as numpy_helper
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set gpu

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
    img = Image.fromarray(image).convert('RGB')
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
def classify_lesions_kor(model,image):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.fromarray(image).convert('RGB')
    inputs = data_transforms(img)
    inputs = inputs.to(device)
    print(inputs.unsqueeze(0).shape)
    outputs = model(inputs.unsqueeze(0))
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    top_p, top_class = probabilities.topk(5, dim=1)
    result = list(zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
    result.sort(key=lambda x: x[1])
    class_names = {
        "0": "아토피 피부염",
        "1": "지루성 피부염",
        "2": "건선",
        "3": "주사",
        "4": "여드름",
    }

    classification_result=[None,None,None,None,None]
    for i,data in enumerate(result):
        class_prob, class_id = data
        class_name = class_names[str(class_id)]
        classification_result[i] = (class_name,class_prob)
    classification_result.sort(key=lambda x:x[1],reverse=True)
    return classification_result
def export_onnx(model_path,image):
    model_name = 'efficientnet-b7'
    # image_size = EfficientNet.get_image_size(model_name)
    # print('Image size: ', image_size)

    # Load model
    model = EfficientNet.from_pretrained(model_name, num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.set_swish(memory_efficient=False)
    model.eval()
    print('Model image size: ', model._global_params.image_size)

    # Dummy input for ONNX
    dummy_input = torch.randn(10, 3, 224, 224)

    # Export with ONNX
    torch.onnx.export(model, dummy_input, "efficientnet-b1.onnx", verbose=False)


    ### dsdf
    onnx_path = "efficientnet-b1.onnx"
    onnx_model = onnx.load(onnx_path)

    # onnx 모델의 정보를 layer 이름 : layer값 기준으로 저장합니다.
    onnx_layers = dict()
    for layer in onnx_model.graph.initializer:
        onnx_layers[layer.name] = numpy_helper.to_array(layer)

    # torch 모델의 정보를 layer 이름 : layer값 기준으로 저장합니다.
    torch_layers = {}
    for layer_name, layer_value in model.named_modules():
        torch_layers[layer_name] = layer_value

        # onnx와 torch 모델의 성분은 1:1 대응이 되지만 저장하는 기준이 다릅니다.
    # onnx와 torch의 각 weight가 1:1 대응이 되는 성분만 필터합니다.
    onnx_layers_set = set(onnx_layers.keys())
    # onnx 모델의 각 layer에는 .weight가 suffix로 추가되어 있어서 문자열 비교 시 추가함
    torch_layers_set = set([layer_name + ".weight" for layer_name in list(torch_layers.keys())])
    filtered_onnx_layers = list(onnx_layers_set.intersection(torch_layers_set))

    difference_flag = False
    for layer_name in filtered_onnx_layers:
        onnx_layer_name = layer_name
        torch_layer_name = layer_name.replace(".weight", "")
        onnx_weight = onnx_layers[onnx_layer_name]
        torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
        flag = compare_two_array(onnx_weight, torch_weight, onnx_layer_name)
        difference_flag = True if flag == True else False

    # ④ onnx 모델에 기존 torch 모델과 다른 weight가 있으면 전체 update를 한 후 새로 저장합니다.
    if difference_flag:
        print("update onnx weight from torch model.")
        for index, layer in enumerate(onnx_model.graph.initializer):
            layer_name = layer.name
            if layer_name in filtered_onnx_layers:
                onnx_layer_name = layer_name
                torch_layer_name = layer_name.replace(".weight", "")
                onnx_weight = onnx_layers[onnx_layer_name]
                torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
                copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
                onnx_model.graph.initializer[index].CopyFrom(copy_tensor)

        print("save updated onnx model.")
        onnx_new_path = os.path.dirname(os.path.abspath(onnx_path)) + os.sep + "updated_" + os.path.basename(onnx_path)
        onnx.save(onnx_model, onnx_new_path)

    # ⑤ 최종적으로 저장된 onnx 모델을 불러와서 shape 정보를 추가한 뒤 다시 저장합니다.
    if difference_flag:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_new_path)), onnx_new_path)
    else:
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)
def check_onnx():
    from onnx import shape_inference
    path = "./efficientnet-b1.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(path)), path)
def compare_two_array(actual, desired, layer_name, rtol=1e-7, atol=0):
    # Reference : https://gaussian37.github.io/python-basic-numpy-snippets/
    flag = False
    try :
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
        print(layer_name + ": no difference.")
    except AssertionError as msg:
        print(layer_name + ": Error.")
        print(msg)
        flag = True
    return flag

class lesion_classification_net:
    def __init__(self, model_path='./best_accuracy_88.85245901639345.pt', model_name='efficientnet-b7'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.device == 'cuda', "cuda가 안되는데요?, classifiaction을 살펴보세요"
        self.num_classes = 5
        self.model = self.load_model(model_path=model_path, model_name=model_name, num_classes=self.num_classes)
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.class_names = {
            "0": "아토피 피부염",
            "1": "지루성 피부염",
            "2": "건선",
            "3": "주사",
            "4": "여드름",
        }

    def load_model(self,model_path,model_name,num_classes):
        model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model.to(device)

    def inference(self, image_path=None, image=None, display=False):
        if (image_path != None):
            image = skimage.io.imread(image_path)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert('RGB')
        inputs = self.data_transforms(image)
        inputs = inputs.to(device)
        outputs = self.model(inputs.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probabilities.topk(5, dim=1)
        result = list(
            zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
        result.sort(key=lambda x: x[1])

        classification_result = [None, None, None, None, None]
        for i, data in enumerate(result):
            class_prob, class_id = data
            class_name = self.class_names[str(class_id)]
            classification_result[i] = (class_name, class_prob)
        classification_result.sort(key=lambda x: x[1], reverse=True)
        if (display):
            print(classification_result)
        return classification_result

class classification_net:
    def __init__(self, Config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Config = Config
        self.Config.num_classes = len(self.Config.class_names)
        self.model = self.__load_model()
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __load_model(self):
        model = EfficientNet.from_pretrained(self.Config.model_name, num_classes=self.Config.num_classes)
        model.load_state_dict(torch.load(self.Config.model_path))
        model.eval()
        return model.to(device)

    def inference(self, image_path=None, image=None, verbose=False):
        if (image_path != None):
            image = skimage.io.imread(image_path)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image).convert('RGB')
        inputs = self.data_transforms(image)
        inputs = inputs.to(device)
        outputs = self.model(inputs.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_p, top_class = probabilities.topk(self.Config.num_classes, dim=1)
        result = list(
            zip([t.item() for t in top_p.squeeze().squeeze()], [t.item() for t in top_class.squeeze().squeeze()]))
        result.sort(key=lambda x: x[1])

        classification_result = [None for _ in range(self.Config.num_classes)]
        for i, data in enumerate(result):
            class_prob, class_id = data
            class_name = self.Config.class_names[str(class_id).zfill(3)]
            classification_result[i] = (class_name, class_prob)
        # print(classification_result)
        classification_result.sort(key=lambda x: x[1], reverse=True)
        if (self.Config.verbose):
            print(classification_result)

        return classification_result[:self.Config.topk]





if __name__ == '__main__':
    # cl_net=lesion_classification_net()
    # import cv2
    # img=cv2.imread('../test_images/rosacea.jpg')
    # cl_net.inference(image_path='../test_images/rosacea.jpg',display=True)
    # cl_net.inference(image=img,display=True)
    # cv2.waitKey(0)
    #
    import skimage.draw
    model = load_model('./best_accuracy_88.85245901639345.pt')
    img_dir = '../test_images'
    dst_dir = '../segmentation/inference_result'
    for i in os.listdir(img_dir):
        print('*'*50)
        img_path = os.path.join(img_dir, i)
        # print(img_path)
        image = skimage.io.imread(img_path)
        dic=classify_lesions_kor(model,image)
        print('\tclassification results',dic)
    #     export_onnx('./best_accuracy_88.85245901639345.pt',image)


