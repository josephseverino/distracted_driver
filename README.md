# distracted_driver

For this project, I built a simple CNN (Convolutional Nueral Network) to predict whether a static image of a driver was distracted while driving. There are 10 different categories for this dataset. All data was dowloaded Kaggle. For more details of dataset see link https://www.kaggle.com/c/state-farm-distracted-driver-detection/data

## Image Processing

The dataset was comprised of over 20,000 train images and nearly 80,000 test images. The images were all the same dimsensions (480 X 640 pixels) in RGB format (color). For processing the images I used OpenCV library to do a few transformations of the data to suit my model and keep the computation expense down.

### Step 1 - Grey Scale

```
img = cv2.imread(f1,0)
```
The 0 here reads in the image as a grey scale image. This process helped my model for two reasons. 
1. It decreased the amount of data processing by 1/3 from its original format (RGB)
2. (opinion) It decreases the noise in data. The colors in the images don't hold any  importance as to whether a driver is ditracted or not and thus I don't want the model to find a possible correlation (i.e. hypothetically, people who were distracted by changing the radio could have been wearing blue jeans more often than not)

### Step 2 - Edge Detection

```
edges = cv2.Canny(img, 50,100)
```
