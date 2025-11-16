# DIT5411-Wong-Po-Yi

### Introduction
This project aims to develop an artificial intelligence software with Chinese Character Recognition. TensorFlow is the main framework for AI software. The number of Chinese characters in the software is 13,065. Its goal is to enhance the performance and user experience of Handwritten Digit Recognition.

### Dataset needed
- Source: Traditional Chinese Handwriting Dataset
- Number of Chinese Characters: 13,165 
- Sample number of every Chinese Character: 50 (40 training + 10 testing)
- 50 images per Chinese character are put in/Traditional_Chinese_Data/cleaned_data folder. They are compressed to Traditional_Chinese_Data.zip and need to use 7-Zip to unzip them.

### Training data - Augmentation Of Images
The following image transformations will be adopted to generate 200 augmented images per Chinese Character.
- Translation: Tx = h / 4.0
               Ty = w / 4.0
               T = np.array([[1, 0, Tx], [0, 1, Ty]], dtype=np.float32)
               Extract 1/4 of the height and width by horizontal and vertical translation. 
               Shift the image to the right by 1/4 of its height and downwards by 1/4 of its width.
- Rotation1: angle = 45
             basePoint = (w / 2.0, h / 2.0)
             rotation_matrix = cv2.getRotationMatrix2D(basePoint, angle, 1.0)
             Rotate 45 degrees clockwise
- Rotation2: angle = -80
             basePoint = (w / 2.0, h / 2.0)
             rotation_matrix = cv2.getRotationMatrix2D(basePoint, angle, 1.0)
             Rotate 80 degrees counterclockwise
- Scaling: None, fx=1.5, fy=1.5
           Enlarge the image's height and width to 1.5 times their original size.
- Shearing: M = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                                augmentedImage = cv2.warpPerspective(imgBeforeAugmentation, M, (int(w * 1.5), int(h * 1.5))
            Enlarge the image by 1.5 times

After augmentation, 200 augmented images per Chinese Character will be generated. Their filenames' format is (Chinese Character)_aug_001 to (Chinese Character)_aug_200. They are put in /Output_Sample/ folder, and then all folders and files are compressed to Output_Sample.zip and need to use 7-Zip to unzip them. 

Traditional_Chinese_Data.zip and Output_Sample.zip
https://vtcmca-my.sharepoint.com/:f:/g/personal/220320080_stu_vtc_edu_hk/Es77JRWxgdpPut4TQn4STw4BtoKyNz5An6sqgGMfJ_Oqqg?e=gQLvdc

### Testing Model
The following AI models based on TensorFlow will be created for the development of Chinese Character Recognition.
1. Input layer - It can input a single-channel (grayscale) image of 128×128 pixels, though writing the code "input_shape=(128, 128, 1)
   First convolutional layer - It uses convolution kernel of 30 sizes of 5×5(Conv2D(30, (5,5)) to extract the local characteristics of images, the size of the output feature map is 124×124×30.
2. First pooling layer - Max pooling of 2×2(writing code "MaxPooling2D(pool_size=(2,2))") is used to downsample the feature map size to 62×62×30.
   Second pooling layer - 15 Convolution kernel of 3×3 (writing code "Conv2D(15, (3,3))"), to output the size feature map to 60×60×15.
   Third pooling layer - 2×2 max pooling again, drop the size of the feature map to 30×30×15.
3. Dropout layer - 20% probability throw neurons (writing code "Dropout (0.2)) to uppress overfitting, which can improve the generalisation of the model
   Flatten layer - Flattening the 3D Feature Map of 30×30×15 to a one-dimensional vector, which can be the input of the fully connected layer
   Fully connected layer
   a. First dense layer - It has 128 neurons that can be used for learning advanced feature combinations (writing code "Dense(128, activation='relu')")
   b. Second dense layer - It has 50 neurons to further abstract features (writing code "Dense(50, activation='relu')")
   c. Output layer - numberOfClasses is the quantity of neurons to output the probability distribution
4. Loss function: It is written code as categorical_crossentropy, which can be used for the labels of multiple classes
   Optimzer - It is written code, as Adam, to balance the convergence speed and effect.
   Evaluation index - It is presented as accuracy to monitor the correct rate of model forecast

### Guideline
1. Download Traditional-Chinese-Handwriting-Dataset.zip from https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset to unzip them.
2. Execute data_deployment_all.ipynb to unzip all_data.zip (which is located on D:\\handwritting_data_all-master\\) to /Traditional_Chinese_Data/cleaned_data.
3. Create the model_train_test.ipynb to write the code about training the model to generate 200 augmented images per Chinese Character
4. Write the code for testing the model.









