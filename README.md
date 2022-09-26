# Recognizing grape breeds based on their leaf images with Deep Convolutional networks

## Introduction
This project was initially a part of my data mining course, but then I found it so exciting and started to read about it and do some research on it by taking advice from my professor [Dr. H Sajedi](https://scholar.google.com/citations?user=YHjV73oAAAAJ&hl=en). So, I tried different things I learned on it, like designing my own network and different methods to get a better result by augmenting data in various ways or exploiting other types of neural networks.

## Methodology
I thinkto make it more easier to follow and faster to search, it is better to explaine what i did in four parts:
- data and data augmentation
- my own model architecture
- pre-trained models
- denoising and autoencoder networks

### Data
First of all, download base data using ```wget``` command and unzip it using ```unzip``` it as below:
```
wget https://www.muratkoklu.com/datasets/Grapevine_Leaves_Image_Dataset.zip
unzip -q Grapevine_Leaves_Image_Dataset.zip 
```

Then, in order to create an out-of-sample set, with a small python script we randomly choosed and moved 20\% of each class to new directory. then our sets was ready to load, loaded them as tensorflow datasets with ``` tf.keras.utils.image_dataset_from_directory ```function.
Then because the image margins was white and white pixles are large in valuse (rgb: 255,255,255) but including no information, i tried to transfer the colors to make the useless white black (rgb: 0,0,0) by map method below:
```
train_data2 = train_data.map(lambda x, y: (255-x, y))
validation_data2 = validation_data.map(lambda x, y: (255-x, y))
test_data2 = test_data.map(lambda x, y: (255-x, y))
```

afterward, since by rotating, flipping, or zooming on an image, its class doenst change, tried to augment the base data with new generated randomly changed images. 

```

layers.RandomFlip("horizontal"),
layers.RandomFlip("vertical"),
layers.RandomZoom(height_factor=(-0.2,0.2), width_factor=(-0.2,0.2),fill_mode='constant', fill_value=0),
layers.RandomRotation(0.3, fill_mode='constant', fill_value=0)
```

although i used these layer inside my architecture to using the true power of randomness, i stored the augmented data in a dataframe to somehow save the gpu processor and time in try and error phases.

### My model
