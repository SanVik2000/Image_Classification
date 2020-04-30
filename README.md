# Image_Classification
A single library for all the popular Image-Classification algorithms<br>
This project is implemented using PyTorch, and each model architecture is defined inside the corresponding .py file. For example, VGG16 architecture is defined inside VGG.py.<br>
## Implemented Models
- LeNet-5
- AlexNet
- VGG16
## Running this Project
To run this project execute the following command<br>
```python3 Main.py```<br>
## Requirements
To run and test this library without any errors, the following requisites should be met:<br>
- Pytorch(With/Without GPU Compatibility)
## Results
The following results have been obtained by using just 20 epochs. More accurate results can be obtained by training it more.<br>
- LeNet-5:<br>
![Training & Testing LeNet-5 on CIFAR10 Dataset](files/LeNet.png)<br>
- AlexNet:<br>
![Training & Testing AlexNet on CIFAR10 Dataset](files/AlexNet.png)<br>
- VGG16:<br>
![Training & Testing VGG16 on CIFAR10 Dataset](files/VGG16.png)<br>
## To-Do
The following algorithms are yet to be implemented and this project is still under constructions.<br>
- [x] AlexNet
- [ ] DenseNet
- [ ] GoogLeNet
- [ ] ResNet
