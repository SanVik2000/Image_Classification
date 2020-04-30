import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
	def __init__(self):
		super(VGG, self).__init__()
		self.conv_layer_1_1 = nn.Conv2d(3, 64, kernel_size = (3,3), padding = 1)
		self.conv_layer_1_2 = nn.Conv2d(64, 64, kernel_size = (3,3), padding = 1)
		self.conv_layer_2_1 = nn.Conv2d(64, 128, kernel_size = (3,3), padding = 1)
		self.conv_layer_2_2 = nn.Conv2d(128, 128, kernel_size = (3,3), padding = 1)
		self.conv_layer_3_1 = nn.Conv2d(128, 256, kernel_size = (3,3), padding = 1)
		self.conv_layer_3_2 = nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1)
		self.conv_layer_3_3 = nn.Conv2d(256, 256, kernel_size = (3,3), padding = 1)
		self.conv_layer_4_1 = nn.Conv2d(256, 512, kernel_size = (3,3), padding = 1)
		self.conv_layer_4_2 = nn.Conv2d(512, 512, kernel_size = (3,3), padding = 1)
		self.conv_layer_4_3 = nn.Conv2d(512, 512, kernel_size = (3,3), padding = 1)
		self.conv_layer_5_1 = nn.Conv2d(512, 512, kernel_size = (3,3), padding = 1)
		self.conv_layer_5_2 = nn.Conv2d(512, 512, kernel_size = (3,3), padding = 1)
		self.conv_layer_5_3 = nn.Conv2d(512, 512, kernel_size = (3,3), padding = 1)
		self.batch_norm_1 = nn.BatchNorm2d(num_features = 64)
		self.batch_norm_2 = nn.BatchNorm2d(num_features = 128)
		self.batch_norm_3 = nn.BatchNorm2d(num_features = 256)
		self.batch_norm_4 = nn.BatchNorm2d(num_features = 512)
		self.max_pooling = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
            #nn.Softmax() ============= This activation can be ommitted because nn.CrossEntropy in Pytorch has Softmax logic built in.....
        )

	def forward(self, x):
		#print("Input : " , x.shape)
		x = F.relu(self.batch_norm_1(self.conv_layer_1_1(x)))
		x = F.relu(self.batch_norm_1(self.conv_layer_1_2(x)))
		x = self.max_pooling(x)
		#print("L1 : " , x.shape)
		x = F.relu(self.batch_norm_2(self.conv_layer_2_1(x)))
		x = F.relu(self.batch_norm_2(self.conv_layer_2_2(x)))
		x = self.max_pooling(x)
		#print("L2 : " , x.shape)
		x = F.relu(self.batch_norm_3(self.conv_layer_3_1(x)))
		x = F.relu(self.batch_norm_3(self.conv_layer_3_2(x)))
		x = F.relu(self.batch_norm_3(self.conv_layer_3_3(x)))
		x = self.max_pooling(x)
		#print("L3 : " , x.shape)
		x = F.relu(self.batch_norm_4(self.conv_layer_4_1(x)))
		x = F.relu(self.batch_norm_4(self.conv_layer_4_2(x)))
		x = F.relu(self.batch_norm_4(self.conv_layer_4_3(x)))
		x = self.max_pooling(x)
		#print("L4 : " , x.shape)
		x = F.relu(self.batch_norm_4(self.conv_layer_5_1(x)))
		x = F.relu(self.batch_norm_4(self.conv_layer_5_2(x)))
		x = F.relu(self.batch_norm_4(self.conv_layer_5_3(x)))
		x = self.max_pooling(x)
		#print("L5 : " , x.shape)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		#print("L6 : " , x.shape)
		x = self.classifier(x)
		#print("Output : " , x.shape)
		return x


