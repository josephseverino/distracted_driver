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
Here, I decided to use openCV's edge detection method. My rationale is that most of information of the different orientations of the driver is represented by detecting the were the outline of the drivers face is oriented or where is hands are. (e.g. facing the passenger or right or left hand with their cellphone). I choose the arguments to be 50 and 100 simply because it is what preserved the most edges of the photo without being too noisy.

### Step 3 - Resize

```
rs = cv2.resize(edges, (int(edges.shape[1]/4), int(edges.shape[0]/4)), interpolation = cv2.INTER_AREA)
```

Due to running into memory allocation errors, I decided to reduce the image down to 120 x 160 pixels which is 1/4 the size of its original format. This made my computation expense 1/4 what it normally be so I could make a more robust architecture.

### Original Image Sample
![Original image sample](https://github.com/josephseverino/distracted_driver/blob/master/img_99.jpg)

### After Processing Image
![Transformed Image](https://github.com/josephseverino/distracted_driver/blob/master/New_Image.jpg)


## Model Architecture

```
def BN_model():

    model = Sequential()
    model.add(Conv2D(16, 3, input_shape=(120, 160, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(16, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

# build the model
model = BN_model()
```

I used a simple CNN with several layers. I used both BatchNormalization and Dropout to mitigate the amount of overfitting the model produced. 

## Results - Accuracy Train/Validation/Test Set

![Accuracy](https://github.com/josephseverino/distracted_driver/blob/master/Results.png)


