# vggNet
> Pytorch implement of vgg, 2 type classification for single channel imgs

### environment
- python 3.7
- cuda 10.2
- cuDNN 8.1.1
- Pytorch(torch、torchvision) 

### packages
- opencv libopencv py-opencv
- numpy
- yaml pyyaml
- matplotlib
- shutil
- os 
### datasets
data files are not uploaded, should be arranged like:<br>
```
./data
  |_/train/
    |_/class 0/
    |_/class 1/
  |_/test/
    |_/class 0/
    |_/class 1/
```
### run model
 ```python
 cd ./run/
 python main.py
 ```
### results
result files are not uploaded<br>
loss and accuray data will be saved in ```./results/train/``` and ```./results/test/```<br>
model file will be saved as ```./results/vgg16.pth.tar/```

### reference
- paper:  [VGG NET](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.1556)
- code refenrence: [pytorch vgg net](https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html)
