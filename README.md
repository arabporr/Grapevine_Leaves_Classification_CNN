# Recognizing grape breeds based on their leaf images with Deep Convolutional networks

## Introduction
This project was initially a part of my data mining course, but then I found it so exciting and started to read about it and do some research on it by taking advice from my professor [Dr. H Sajedi](https://scholar.google.com/citations?user=YHjV73oAAAAJ&hl=en). So, I tried different things I learned on it, like designing my own network and different methods to get a better result by augmenting data in various ways or exploiting other types of neural networks.

> **Note:** If you had problem in opening any of the ```.ipynb``` files, I exported them as ```.pdf``` files available in ["pdf_files" directory](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/tree/main/pdf_files).

## Methodology
I think to make it easier to follow and faster to search; it is better to explain what I did in four parts:
- Data and data augmentation
- My own model architecture
- Pre-trained models
- Denoising and autoencoder networks

### Data
First of all, download base data using ```wget``` command and unzip it using ```unzip``` it as below:
```
wget https://www.muratkoklu.com/datasets/Grapevine_Leaves_Image_Dataset.zip
unzip -q Grapevine_Leaves_Image_Dataset.zip 
```
- Here is a sample of the base data:

![base_data](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/19f152ac4e8d782e7c1ade6fec6bcb3ce843a540/readme_images/base_data.png)

Then, to create an out-of-sample set with a small python script, we randomly chose and moved 20\% of each class to the new directory. Then our data was ready to load, loaded as TensorFlow datasets with ``` tf.keras.utils.image_dataset_from_directory ``` function.
Then because the margins of the images are white and white pixels are large in value (RGB code: 255,255,255) but contain no information, I tried to transfer the colors to turn the useless white into black (RGB code: 0,0,0) by the map method below:
```
train_data2 = train_data.map(lambda x, y: (255-x, y))
validation_data2 = validation_data.map(lambda x, y: (255-x, y))
test_data2 = test_data.map(lambda x, y: (255-x, y))
```

Afterward, since rotating, flipping, or zooming on an image, its class does not change; I tried to augment the base data with newly generated randomly changed images. 
```
layers.RandomFlip("horizontal"),
layers.RandomFlip("vertical"),
layers.RandomZoom(height_factor=(-0.2,0.2), width_factor=(-0.2,0.2),fill_mode='constant', fill_value=0),
layers.RandomRotation(0.3, fill_mode='constant', fill_value=0)
```
- Here is a sample of the augmented transformed data:

![augmented_transformed_data](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/cd56a8cc8f3b62388f2f54701bfe37d810a01495/readme_images/transformed_data.png)

Although I used these layers inside my architecture to use the true power of randomness, I stored simple augmented data in a dataset to somehow save the GPU processor and time in the try and error phases.

### My architecture
In this part which is available [here!](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/f16cf69a86498c3d848cabb5ef6b38390a61f354/My_Own_Model.ipynb); What I did was creating a model starting with 3 data augmentation layers to prevent from overfitting and also provide better learning, then 12 convolution and pooling layers to extracting every little information, and finally after flattening, five dense layers were in charge of classification. I used a bunch of different architectures and changed each one many times to end up with this result, which is good enough to be compared with famous networks on this data. 

- A more detailed summary of the model is shown below:

![Model_Summary](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/f16cf69a86498c3d848cabb5ef6b38390a61f354/readme_images/My_architecture.png)

Afterward, in the training phase, I used the ```adam``` optimizer and ```SparseCategoricalCrossentropy``` loss function to train the network for 200 epochs and a batch size of 32. The accuracy and loss during the training is provided below:
- accuracy curve

![My_model_acc](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/2162b4dc8e896047749c0abaf9db3ee6e2273ecc/readme_images/My_model_train_acc.png)

- loss curve

![My_model_loss](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/2162b4dc8e896047749c0abaf9db3ee6e2273ecc/readme_images/My_model_train_loss.png)

In the end, I tried to test the model with the unseen out-of-sample data to see whether the results were real or not [overfitting]. For this test, we show the model 100 images [20 from each class] and check the predicted class with the real one. The result in the table below shows great work, and a good thing to be mentioned is <ins>due to the confusion matrix, the learning was not biased</ins>, which is very important in classification tasks.
- Result table:

![My_model_res](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/f91bedcb29a8801fbc670513dabc1420ffacdef7/readme_images/My_model_result.png) 

### Pre-trained models
In this part, which codes are available [here!](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/4474829f94c6067b1f785c3e352862b33e7ab7ff/Pre-Trained_Models.ipynb), I tried different models including ```Xception```, ```VGG16```, ```VGG19```, ```ResNet50```, ```ResNet101```, ```ResNet152```, ```InceptionV3```, and ```InceptionResNetV2``` in the same structure in order to find the best model. Consequently, test it with different seeds and compare its results with my model.

The architecture I used starts with three Keras data augmentation layers in which the input data is randomly rotated, flipped, or zoomed, then the model itself has been placed, and finally, three dense layers for classifying the model's output into our desired five classes.

- As an example, you can see more details for the VGG16 model summary below:

![Pre_trained_arc_vgg16](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/caf031eb519eb3d06b71e112b90bc7337dff9cac/readme_images/Pre_trained_VGG16_arc.PNG) 

You can find the codes, accuracy, and loss curves for each model in the notebook with more details. Also below, you can see and compare all of them at once:

- Pre-trained models <ins>accuracy on training data during training phase</ins>:

![Pre_trained_train_acc](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/d2ff8940ee9fc386b239ca71bcb8cba21b2cfd13/readme_images/Pre_trained_train_acc.png)


- Pre-trained models <ins>accuracy on validation data during training phase</ins>:

![Pre_trained_val_acc](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/d2ff8940ee9fc386b239ca71bcb8cba21b2cfd13/readme_images/Pre_trained_val_acc.png)

To sum up, what I found out was that ```Xception```, ```InceptionV3```, and ```InceptionResNetV2``` was so bad and weren't even close to the others. But on the other hand, both the VGG and ResNet networks worked quite well and ended up with accuracies of around 80 percent. However, The best model was ```ResNet152```, which reached 84\% on unseen out-of-sample data!

- The chart below compared the accuracy of pre-trained models on out-of-sample data:

![Pre_trained_results](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/42c7c7172d2dff170d050420e0eb2eaabc690e8c/readme_images/Pre_trained_resualts.png)

### Exploiting denoising and autoencoder networks
The idea of this part (codes are available [here!](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/cd3122c2b9531cd72b5eefdc7f9d0946800543ac/Denoising_And_Encoding_Networks.ipynb)) was to create a model in which the important parts of images are found and emphasized before the image goes into our classifier part of the network, so maybe this makes the model focus on essential pieces of information only and provides us a better result. Which unfortunately didn't happen.

For the denoising part, I firstly created a noisy version of each image in the augmented dataset with a noise factor of 0.3 and a mean of 127. Afterward, I trained my network with these images for 30 epochs. Even though I spent a week working on it, I couldn't manage to get a good result. The model summary and a sample of its output are presented below:

- Denoising model summary:

![Denoising_architecture](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/80269548473e115e3be2744efd1c1b888202f706/readme_images/Denoising_arc.png)


- Denoising sample output: 

![Denoising_sample](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/80269548473e115e3be2744efd1c1b888202f706/readme_images/Denoising_sample.png)

> images in the first row are the noisy ones, and the second row is the network's output from the same image in row one. The last row is the actual image itself.


After creating this denoising network, it was time to attach it to the rest of the network's body. Since we have previously seen the performance of the ```ResNet152```, I decided to use a combination of all I got to get the best I can. 
- The final architecture was this : 

![DenoisingResNet152_arc](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/80269548473e115e3be2744efd1c1b888202f706/readme_images/DenoisingResNet152_arc.png)

> The layer sequential_10 is the previously trained denoising network.

After all of this, I trained the model with 100 epochs of data, and what I got was this accuracy curve during the training phase, which shows us that the model learning converged on about 70\%.

- The accuracy curve:

![Denoising_train_acc](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/80269548473e115e3be2744efd1c1b888202f706/readme_images/Denoising_train_acc.png)


- The results table:

![Denoising_result](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/80269548473e115e3be2744efd1c1b888202f706/readme_images/Denoising_result.png)

For the autoencoder part, the situation was much worse, and despite my endeavor to make a good network, the best model's output looked like a faded purple circle which clearly missed so much information.

- Autoencoder output:

![Autoencoder_output](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/fbf7dc007b70d2cac90a05903b06eafae14902ee/readme_images/Autoencoder.png)

> The images in the first row are the actual images, and the images in the second row are the reconstructed form of the encoded version of the ones above.

With what we see above, it is expected not to get a good result. Like the denoising part, I used the ```ResNet152``` here too. For the report's sake, I bring the results below, although its accuracy is just a little bit more than random answering.

The results table:

![AutoencoderResNet_result](https://github.com/arabporr/Grapevine_Leaves_Classification_CNN/blob/fbf7dc007b70d2cac90a05903b06eafae14902ee/readme_images/AutoencoderResNet_result.png) 

## References
1. [A CNN-SVM study based on selected deep features for grapevine leaves classification](https://www.sciencedirect.com/science/article/abs/pii/S0263224121013142?via%3Dihub)

2. [Image denoising method based on a deep convolution neural network](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-ipr.2017.0389)
