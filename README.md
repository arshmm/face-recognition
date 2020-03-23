# Face Recognition

Face recognition using OpenCV and python

### Prerequisites

First download anaconda into your system and create a virtual enviroment

```
conda create --name yourenvname python=3.7
```

### Installing

Instructions for installing numpy and OpenCV

First we will activate the virtual enviroment in cmd 

```
conda activate yourenvname 
```
then install numpy 
```
conda install -c anaconda numpy
```
then install OpenCV and OpenCV contrib 
```
conda install -c conda-forge opencv
conda install -c michael_wild opencv-contrib
```

check wheather the libraries are properly installed
for that write the following lines in cmd
```
python
import cv2
print(cv2.__version__)
import numpy
print(numpy.__version__)
```

## Running the tests

Run Tester.py script on commandline to train recognizer on training images and also predict test_img:<br>
##### python tester.py
1.Place some test images in test_images folder that you want to predict  in tester.py file</br>
2.Place Images for training the classifier in training_images folder.If you want to train clasifier to recognize multiple people then add each persons folder in separate label markes as 0,1,2,etc and then add their names along with labels in tester.py/videoTester.py file in 'name' variable.</br>
3.To generate test images for training classifier use videoimg.py file.</br>
4.To do test run via tester.py give the path of image in test_img_path variable</br>
5.Use "videoTester.py" script for predicting faces realtime via your webcam.(But ensure that you run tester.py first since it generates td.yml file that is being used in "videoTester.py" script.


