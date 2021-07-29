## 🚀 Facemask Detection - MobileNetV2 CNN Architecture 🚀
<img src="/images/3.png" width="100%" height="50%"/>
<img src="/images/1.png" width="100%" height="50%"/>
<img src="/images/2.png" width="100%" height="50%"/>

### 1. Makesure you already have libraries
#### Create virtualenv
```bash
virturalenv env
```
#### Activate env
```bash
source env/Scripts/activate
```

### Install dependencies
```bash
pip install -r resources.txt
```
### 2. Train Model using CUDA and CUDNN (if you have GPU)
#### Train Model
```bash
python train.py --dataset dataset
```
#### Detect Video Streaming
```bash
python detect.py 
```
## :+1: Credits
* [https://www.pyimagesearch.com/](https://www.pyimagesearch.com/)
* [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)