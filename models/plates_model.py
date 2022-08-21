import sys
sys.path.append('../')
import torch
import torch.nn as nn
from PIL import Image
from models.crnn import CRNN
from common.ctc_decoder import ctc_decode
import numpy as np
from torchvision.transforms.functional import crop, resize, rgb_to_grayscale
from models.models import yolo_small
import cv2
import matplotlib.pyplot as plt

CHARS = '0123456789АВЕКМНОРСТУХ'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

yolo_checkpoint = '../checkpoints/plates_sd.pt'
crnn_checkpoint = '../checkpoints/crnn_014000_loss1.8461857752828725.pt'#crnn_012000_loss1.6033634845448912.pt'


common_config = {
    'data_dir': '../datasets/russian_plates',
    'img_width': 256,
    'img_height': 64,
    'map_to_seq_hidden': 64,
    'rnn_hidden': 256,
    'leaky_relu': False,
}


class PlatesModel(nn.Module):
	def __init__(self):
		super(PlatesModel, self).__init__()
		self.num_class = len(LABEL2CHAR) + 1
		self.img_width = common_config['img_width']
		self.img_height = common_config['img_height']

		#self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_checkpoint)  # local model
		self.yolo_model = yolo_small(weights_path=yolo_checkpoint)

		self.crnn = CRNN(1, self.img_height, self.img_width, self.num_class,
						 map_to_seq_hidden=common_config['map_to_seq_hidden'],
						 rnn_hidden=common_config['rnn_hidden'],
						 leaky_relu=common_config['leaky_relu'])

		self.device = 'cuda' if next(self.crnn.parameters()).is_cuda else 'cpu'
		self.crnn.load_state_dict(torch.load(crnn_checkpoint, map_location=self.device))
		self.crnn.to(self.device)

	def forward(self, imgs):
		"""
		:param imgs: tensor [N, 3, W, H]
		"""
		det = self.yolo_model(imgs)
		imgs = rgb_to_grayscale(imgs)
		for i in range(len(imgs)):
			img = imgs[i] # torch tensor for i-th image
			bbset = det[i]['boxes']
			conf = det[i]['scores']

			true_conf = conf > 0.25

			
			conf = conf[true_conf]
			bbset = bbset[true_conf]

			platesset = [] # set of plates for a single frame
			for bbox in bbset:
				x, y, x2, y2 = bbox.tolist()
				h = y2 - y
				w = x2 - x
				cr = crop(img, int(y), int(x), int(h), int(w))
				cr = resize(cr, [self.img_height, self.img_width]).unsqueeze(0)
				#cr = (cr / 127.5) - 1.0
				#cr_save = cr.permute(1, 2, 0).cpu().detach().numpy() 
				#plt.imsave('temp.png', cr_save)
				with torch.no_grad():
					cr = cr.to(self.device)
					logits = self.crnn(cr)
					log_probs = torch.nn.functional.log_softmax(logits, dim=2)
					preds = ctc_decode(log_probs, method='beam_search', beam_size=10,
						label2char=LABEL2CHAR)
					platesset.append(''.join(preds[0]))

			det[i]['labels'] = platesset
			det[i]['scores'] = conf
			det[i]['boxes'] = bbset

		return det