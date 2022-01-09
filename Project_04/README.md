[//]: # (Image References)

[image1]: ./apps/data/images/sample_dog_output.png "Sample Output"
[image2]: ./apps/data/images/vis_label_distribution.png "VGG-16 Model Keras Layers"
[image3]: ./apps/data//images/load_dataset.png "VGG16 Model Figure"
[image4]: ./apps/data//images/load_dataset_human.png "VGG16 Model Figure"
[image5]: ./apps/data//images/custom_model.png "VGG16 Model Figure"
[image5]: ./apps/data//images/compile_and_train.png "VGG16 Model Figure"
## Table of Contents
1. Project Overview
2. Project Instructions
   1. Requirements Packages
   2. Data Visualization And Analyzation
   3. Re-train Model
   4. Demo Webapp
3. File Descriptions

## 1. Project Overview

In this project, I will build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  
Given an image of a dog, my algorithm will identify an estimate of the canine’s breed.  
If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

## 2. Project Instructions
### i. Requirements Packages
	opencv-python==3.2.0.6
	h5py==2.6.0
	matplotlib==2.0.0
	numpy==1.12.0
	scipy==0.18.1
	tqdm==4.11.2
	keras==2.0.2
	scikit-learn==0.18.1
	pillow==4.0.0
	ipykernel==4.6.1
	tensorflow==1.2
	streamlit>=1.0.0
### ii. Data Analyzation
##### Download links:
1. [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).
2. [dog breed dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
3. [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz).

#### Data Analyzation
There are **_133_** total dog categories.
There are **_8351_** total dog images.
Of which the most numerous dog breeds in the data are the **_Alaskan Malamute_** with **_96_** images, and the least are the **_Norwegian Buhund_** and **_Xoloitzcuintli_** with **_33_** images.

Observing the **_Label Distribution_** chart, it can be seen that the label is distributed relatively evenly, but the difference in ratio is still quite large. Augmentation algorithms can be used to increase the amount of data, and at the same time dividing the train and test sets proportionally also helps to better control the model.

![Label Distribution][image2]
### iii. Re-train model
1. Open ```base_workspace/dog_app.ipynb```.
2. Download [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features and save to ```base_workspace/bottleneck_features```
3. Replace the path to dog breed dataset and human dataset
![dog breed dataset][image3]
![human dataset][image4]
4. Update the model as you want, remember that do not change the last Dense layer.
![Update the model][image5]
5. Compile and train the model, you can change the epochs, batch_size or callback as you want.
![Compile and train the model][image6]
### iv. Demo webapp
1. Copy or download [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features to ```apps/data/bottleneck_features```
2. If you have new re-trained model:
   1. Copy your re-trained model to ```apps/data/saved_models```
   2. Change the you re-trained model name in ```model_builder``` function on ```apps/main.py```
3. Run the following        
   ```
   cd apps
   streamlit run app.py --server.port 8080
   ```
4. Access to the url http://localhost:8080 or [http://<server-extenal-ip>:8080]() and try it!
## 3. File Descriptions
```
│   README.md
│
├───apps
│   │   app.py
│   │   CODEOWNERS
│   │   extract_bottleneck_features.py
│   │   main.py
│   │
│   └───data
│       │   dog_names.txt
│       │
│       ├───bottleneck_features
│       │
│       ├───haarcascades
│       │       haarcascade_frontalface_alt.xml
│       │
│       ├───images
│       │       American_water_spaniel_00648.jpg
│       │       Brittany_02625.jpg
│       │       Curly-coated_retriever_03896.jpg
│       │       Labrador_retriever_06449.jpg
│       │       Labrador_retriever_06455.jpg
│       │       Labrador_retriever_06457.jpg
│       │       sample_cnn.png
│       │       sample_dog_output.png
│       │       sample_human_2.png
│       │       sample_human_output.png
│       │       Welsh_springer_spaniel_08203.jpg
│       │
│       ├───saved_models
│       │       .gitignore
│       │       weights.best.from_scratch.hdf5
│       │       weights.best.Resnet50.hdf5
│       │       weights.best.VGG16.hdf5
│       │
│       └───temp
│
├───base_workspace
│   │   bottleneck_features.zip
│   │   CODEOWNERS
│   │   dog_app-zh.ipynb
│   │   dog_app.ipynb
│   │   extract_bottleneck_features.py
│   │   LICENSE.txt
│   │   README.md
│   │
│   ├───bottleneck_features
│   │       .gitignore
│   │
│   ├───haarcascades
│   │       haarcascade_frontalface_alt.xml
│   │
│   ├───images
│   │       American_water_spaniel_00648.jpg
│   │       Brittany_02625.jpg
│   │       Curly-coated_retriever_03896.jpg
│   │       Labrador_retriever_06449.jpg
│   │       Labrador_retriever_06455.jpg
│   │       Labrador_retriever_06457.jpg
│   │       sample_cnn.png
│   │       sample_dog_output.png
│   │       sample_human_2.png
│   │       sample_human_output.png
│   │       Welsh_springer_spaniel_08203.jpg
│   │
│   ├───requirements
│   │       dog-linux-gpu.yml
│   │       dog-linux.yml
│   │       dog-mac-gpu.yml
│   │       dog-mac.yml
│   │       dog-windows-gpu.yml
│   │       dog-windows.yml
│   │       requirements-gpu.txt
│   │       requirements.txt
│   │
│   └───saved_models
│           .gitignore
│           weights.best.from_scratch.hdf5
│           weights.best.Resnet50.hdf5
│           weights.best.VGG16.hdf5
```