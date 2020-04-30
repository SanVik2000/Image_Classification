import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet,self).__init__()
		self.conv_layer_1 = nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2)
		self.conv_layer_2 = nn.Conv2d(64, 192, kernel_size = 5, padding = 2)
		self.conv_layer_3 = nn.Conv2d(192, 384, kernel_size = 3, padding = 1)
		self.conv_layer_4 = nn.Conv2d(384, 256, kernel_size = 3, padding = 1)
		self.conv_layer_5 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1)
		self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
		self.avg_pool = nn.AdaptiveAvgPool2d((6,6))
		self.classifier = nn.Sequential(
			nn.Linear(256*6*6, 4096),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, 10)
			)

	def forward(self, x):
		#print("Input Shape : " , x.shape)
		x = F.relu(self.conv_layer_1(x))
		x = self.max_pool(x)
		#print("L1 : " , x.shape)
		x = F.relu(self.conv_layer_2(x))
		x = self.max_pool(x)
		#print("L2 : " , x.shape)
		x = F.relu(self.conv_layer_3(x))
		#print("L3 : " , x.shape)
		x = F.relu(self.conv_layer_4(x))
		#print("L4 : " , x.shape)
		x = F.relu(self.conv_layer_5(x))
		#print("L5 : " , x.shape)
		x = self.max_pool(x)
		#print("L6 : " , x.shape)
		x = self.avg_pool(x)
		#print("L7 : " , x.shape)
		x = x.view(x.size(0), -1) # Flatten Layer
		#print("L8 : " , x.shape)
		x = self.classifier(x)
		#print("L9 : " , x.shape)
		return x


