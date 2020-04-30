import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import time

from LeNet import LeNet
from VGG import VGG
from AlexNet import AlexNet
import tqdm as tqdm

def Print_Line():
	STR = []
	Length = os.get_terminal_size().columns
	for i in range(0,Length):
		STR.append("=")
	STR = ''.join(STR)
	print(STR)
	return 0


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat', 'deer',
	           'dog', 'frog', 'horse', 'ship', 'truck')

def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 

def Train_Model(model, trainloader, model_name):
	Loss = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


	start_gpu_time = time.time()

	

	print("\033[93m==>\033[00m Starting Training using \033[93m", model_name, "\033[00m:")

	for epoch in range(20):

		correct = 0
		Total_Samples = 0
		running_loss = 0.0
		start_epoch_time = time.time()

		for i, data in enumerate(trainloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)

			optimizer.zero_grad()

			outputs = model(inputs)
			loss = Loss(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * inputs.size(0)
			Total_Samples += labels.size(0)

			_, predicted = torch.max(outputs,1)
			correct += predicted.eq(labels).sum().item()

		end_epoch_time = time.time()

		print("Epoch :\033[93m %2.0f \033[00m::::: Loss :\033[91m % 5.5f \033[00m::::: Accuracy :\033[92m % 5.5f \033[00m::::: Epoch Time :\033[34m % 5.5f \033[00m" %(epoch+1,(running_loss / Total_Samples),(correct/Total_Samples),(end_epoch_time-start_epoch_time)))

	PATH = "Classification_" + model_name + ".pth"
	torch.save(model.state_dict(), PATH)

	print("Elapsed Time : " , time.time() - start_gpu_time)

def Test_Model(model, testloader, model_name) :

	print("\033[93m==>\033[00m Evaluating using \033[93m", model_name, "\033[00m:")

	correct = 0
	total = 0
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        images = images.to(device)
	        labels = labels.to(device)
	        outputs = model(images)
	        _, predicted = torch.max(outputs.data, 1)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: \033[92m%d %%\033[00m' % (
	    100 * correct / total))

	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
	    for data in testloader:
	        images, labels = data
	        images = images.to(device)
	        labels = labels.to(device)
	        outputs = model(images)
	        _, predicted = torch.max(outputs, 1)
	        c = (predicted == labels).squeeze()
	        for i in range(4):
	            label = labels[i]
	            class_correct[label] += c[i].item()
	            class_total[label] += 1


	for i in range(10):
	    print('Accuracy of %5s : \033[92m%2d %%\033[00m' % (
	        classes[i], 100 * class_correct[i] / class_total[i]))

def Prepare_Data(image_size):
	# Data
	print('\033[93m==>\033[00m Preparing data...')
	transform_train = transforms.Compose([
	    transforms.Resize(image_size),
	    transforms.ToTensor(),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	transform_test = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])

	trainset = torchvision.datasets.CIFAR10(
	    root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(
	    trainset, batch_size=128, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(
	    root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(
	    testset, batch_size=100, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer',
	           'dog', 'frog', 'horse', 'ship', 'truck')

	return trainloader, testloader, classes


def main():

	print("\033[37mSanVik2000".center(os.get_terminal_size().columns))
	print("\033[37mLibrary of Famous Classification Algorithms".center(os.get_terminal_size().columns))
	print("\033[37mImplemented in PyTorch".center(os.get_terminal_size().columns))

	Print_Line()


	print("\033[93mSOFTWARE DETAILS\033[00m".center(os.get_terminal_size().columns))
	print("\033[00m================\033[00m".center(os.get_terminal_size().columns))
	print("Python Version       : \033[93m" , sys.version)
	print("\033[00mPyTorch Verison      : \033[93m", torch.__version__)
	print("\033[00mPyTorch using Device : \033[93m" , device, "\033[00m")

	Print_Line()

	print("Choose from the following models to classify images using the CIFAR-10 Dataset\033[00m")
	print("*\033[93m 1 \033[00m* LeNet")
	print("*\033[93m 2 \033[00m* AlexNet")
	print("*\033[93m 3 \033[00m* VGG16")
	model = int(input("Enter model choice : "))
	if model == 1 :
		model = LeNet()
		model_name = "LeNet"
		image_size = 32
	if model == 2 :
		model = AlexNet()
		model_name = "AlexNet"
		image_size = 256
	if model == 3 :
		model = VGG()
		model_name = "VGG16"
		image_size = 32

	Print_Line()

	trainloader, testloader, classes = Prepare_Data(image_size)

	
	model = model.to(device)

	Train_Model(model, trainloader, model_name)
	Test_Model(model, testloader, model_name)

main()