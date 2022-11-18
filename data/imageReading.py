import splitfolders
from tensorflow import keras
import shutil
import os

def createFolders(folderName, train_split, val_split, test_split, delete, seed):
    output_path = folderName + '_split'
    if(os.path.exists(output_path) and delete):
        print("Delete old data folder: " + output_path)
        shutil.rmtree(output_path)
    splitfolders.ratio(folderName, output=output_path, seed=seed, ratio=(train_split, val_split, test_split), group_prefix=None, move=False)
    return output_path

def readData(folderName, image_size, batch_size, preprocess_input, seed=None, split_folder=True, delete=True, train_split=0.8, val_split=0.1, test_split=0.1):
    if(split_folder):
        ds_path = createFolders(folderName, train_split, val_split, test_split, delete, seed)
    else:
        ds_path = folderName

    train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    
    train_path = ds_path + '/train'
    val_path = ds_path + '/val'
    test_path = ds_path + '/test'

    train_batches = train_gen.flow_from_directory(
        train_path,
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        color_mode="rgb",
        seed=seed
    )

    val_batches = valid_gen.flow_from_directory(
        val_path,
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True,
        color_mode="rgb",
        seed=seed
    )

    test_batches = test_gen.flow_from_directory(
        test_path,
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )
    return (train_batches, val_batches, test_batches)
