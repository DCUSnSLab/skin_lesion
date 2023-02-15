from ultralytics import YOLO
import numpy as np
import cv2

from setproctitle import *
import os

def draw_box(image,boxes,model_names, label=True):
	for i, box in enumerate(boxes.boxes):
		lw = max(round(sum(image.shape) / 2 * 0.003), 2)
		p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
		cv2.rectangle(image, p1, p2, (0,0,255), thickness=lw, lineType=cv2.LINE_AA)
		# print(int(boxes.cls[i]),model_names[int(boxes.cls[i])])
		cur_class = model_names[int(boxes.cls[i])]

		if label:
			cur_label = '' + cur_class
			tf = max(lw - 1, 1)  # font thickness
			w, h = cv2.getTextSize(cur_label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
			outside = p1[1] - h >= 3
			p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
			cv2.rectangle(image, p1, p2, (0,0,255), -1, cv2.LINE_AA)  # filled
			cv2.putText(image, cur_label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, (255, 255, 255),
			            thickness=tf, lineType=cv2.LINE_AA)

	return image
def draw_segmentation(image,segmentations,alpha=0.8,show=False):
	for segmentation in segmentations:
		segmentation = segmentation.cpu().numpy()
		segmentation = cv2.resize(segmentation, (image.shape[1], image.shape[0]))
		color = np.array([0, 0, 255], dtype='uint8')
		masked_img = np.where(segmentation[..., None], color, image)
		image = cv2.addWeighted(src1=image, alpha=alpha, src2=masked_img, beta=0.2, gamma=0)
	if(show):
		cv2.imshow("image",image)
		cv2.waitKey(30000)

	return image



def get_images_paths(image_path):
	if (os.path.isfile(image_path)):
		return [image_path]
	elif (os.path.isdir(image_path)):
		file_paths = [x for x in os.listdir(image_path)]
		# print(os.listdir(image_path))
		for i, file in enumerate(file_paths):
			if ('.png' in file or '.jpg' in file):
				pass
			else:
				file_paths[i] = None
		temp_indexes = list(np.where(np.array(file_paths) != None)[0])
		file_paths = [os.path.join(image_path,file_paths[x]) for x in temp_indexes]
		if(len(file_paths)>0):
			return file_paths
		else:
			raise Exception("No valid image files found, please check dir")
	else:
		raise Exception("image_path is not dir or valid image, please check image_path")
		# return file_paths

# def run_single_image(image_path,model):
#
def get_results(image_path,model,verbose,device,label):
	model_names = model.names
	image = cv2.imread(image_path)
	results = model.predict(image_path, device=device, verbose=verbose)
	length = len(results[0].boxes.cpu().numpy())
	if (length <= 0):
		print('\033[31m' + "object is not detected!!!", "image_path: " + image_path + '\033[0m')
		# raise Exception('\033[31m' + "object is not detected!!!", "image_path: " + image_path + '\033[0m')
	else:
		for i, result in enumerate(results):
			image = draw_box(image=image, boxes=result.boxes, model_names=model_names, label=label)
			image = draw_segmentation(image, segmentations=result.masks.masks)

	return image


def inference(image_path, model_path, show=False, save=False,verbose=False,device=1,label=True):
	model = YOLO(model_path)
	image_path_list = get_images_paths(image_path)
	for image_path in image_path_list:
		image = get_results(image_path=image_path, model=model,verbose=verbose,device=device,label=label)
		if (save):
			raise NotImplementedError("save function is not implemented ㅜㅜ")
		if (show):
			cv2.imshow("image", image)
			cv2.waitKey(1000)


if __name__ == '__main__':
	setproctitle('yolo test')

	# image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/KakaoTalk_20221031_182210929.png"
	image_path = "/home/dgdgksj/skin_lesion/ultralytics/test_images/"

	model_path = "/home/dgdgksj/skin_lesion/ultralytics/best_n.pt"
	# model_path = "yolov8n-seg.pt"
	inference(image_path=image_path,model_path=model_path,show=True,save=False,verbose=False,device=1,label=False)
	# print(get_images_paths(image_path))


