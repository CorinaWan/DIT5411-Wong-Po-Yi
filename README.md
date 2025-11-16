# DIT5411-Wong-Po-Yi

Introduction
This project aims to develop an artificial intelligence software with Chinese Character Recognition. Tensorflow is the main framework of the AI software. The number of Chinese characters in the software is 13,065. Its goal is to enhance the performance and user experience of Handwritten Digit Recognition.

Dataset needed
- Source: Traditional Chinese Handwriting Dataset
- Number of Chinese Characters: 13,165 
- Sample number of every Chinese Character: 50 (40 training + 10 testing)
- 50 images per Chinese Characters are put on /Traditional_Chinese_Data/cleaned_data folder. They are compressed to Traditional_Chinese_Data.zip.001 to Traditional_Chinese_Data.zip.116 (116 zip files totally) and need to use 7-Zip to unzip them.

Training data - Augmentation Of Images
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

After augmentation, 200 augmented images per Chinese Character will be generated. Their filenames' format is (Chinese Character)_aug_001 to (Chinese Character)_aug_200. They are put on /Output_Sample/ folder and then all folders and files are compressed to Output_Sample.zip.0001 to Output_Sample.zip.1033 (1033 zip files totally) and need to use 7-Zip to unzip them. 

Testing Model















