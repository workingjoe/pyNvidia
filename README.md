# pyNvidia
Jetson Nano Inference experiments

# blindly following [AI on the Jetson Nano](https://www.youtube.com/watch?v=5rbOsKCZ-VU&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&index=49) 
---
* This is very likely cloned from [Dusty](https://github.com/dusty-nv/jetson-inference) README.md file ...
---
* sudo apt-get install git cmake libpython3-dev python3-numpy 
* cd ~/Downloads
* git clone --recursive https://github.com/dusty-nv/jetson-inference 
* cd jetson-inference
* mkdir build
* cd build
* cmake ../

* When presented with selection dialog boxes,
* Select ALL pre-trained Image Recognition models (2.2GB)
* Select ALL detection models (395MB) 
* Take SEMANTIC Segmentation 24.25,27,29,31,33 (not legacy models)
* Take ALL Image-Processing -ALL models

* Also -- select PyTorch 3.X version for installation

* make -j$(nproc)
* sudo make install
* sudo ldconfig
* sudo apt-get install v4l-utils

---
* test with loading Python3  and 'import torch'  'import torchvision'  and they should load without error
* quit()
---

create quick test program with 
* n_net = jetson_inference.detectNet('ssd-mobilenet-v2', threshold=0.5)
* n_net = jetson_inference.imageNet('googlenet')

* 
