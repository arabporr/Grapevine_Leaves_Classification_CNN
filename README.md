# Recognizing grape breeds based on their leaf images with Deep Convolutional networks

## Introduction
This project was initially a part of my data mining course, but then I found it so exciting and started to read about it and do some research on it by taking advice from my professor [Dr. H Sajedi](https://scholar.google.com/citations?user=YHjV73oAAAAJ&hl=en). So, I tried different things I learned on it, like designing my own network and different methods to get a better result by augmenting data in various ways or exploiting other types of neural networks.

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

Then, to create an out-of-sample set with a small python script, we randomly chose and moved 20\% of each class to the new directory. Then our data was ready to load, loaded as TensorFlow datasets with ``` tf.keras.utils.image_dataset_from_directory ``` function.
Then because the margins of the images are white and white pixels are large in value (RGB code: 255,255,255) but contain no information, I tried to transfer the colors to turn the useless white into black (RGB code: 0,0,0) by the map method below:
```
train_data2 = train_data.map(lambda x, y: (255-x, y))
validation_data2 = validation_data.map(lambda x, y: (255-x, y))
test_data2 = test_data.map(lambda x, y: (255-x, y))
```

Afterward, since rotating, flipping, or zooming on an image, its class doesn't change; I tried to augment the base data with newly generated randomly changed images. 
```
layers.RandomFlip("horizontal"),
layers.RandomFlip("vertical"),
layers.RandomZoom(height_factor=(-0.2,0.2), width_factor=(-0.2,0.2),fill_mode='constant', fill_value=0),
layers.RandomRotation(0.3, fill_mode='constant', fill_value=0)
```

Although I used these layers inside my architecture to use the true power of randomness, I stored simple augmented data in a dataset to somehow save the GPU processor and time in the try and error phases.

### My model
