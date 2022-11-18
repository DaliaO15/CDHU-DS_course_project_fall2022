import splitfolders
from tensorflow import keras
import shutil
import os


def createFolders(folderName, train_split, val_split, test_split, delete, seed):
    '''
    Create directory "folderName_split" with sub directories "train", "val", "test"
    The files are copied from "folderName" to "folderName_split" 
    
    :param folderName: Name of original folder
    :param train_split: Size of train data [0, 1]
    :param val_split: Size of validation data [0, 1]
    :param test_split: Size of test data [0, 1]
    :param delete: If True overwrite folder "folderName_split" 
    :param seed: Seed for data split
    
    :return Path of newly created directory
    '''
    
    output_path = folderName + '_split'
    if(os.path.exists(output_path) and delete):
        print("Delete old data folder: " + output_path)
        shutil.rmtree(output_path)
    splitfolders.ratio(folderName, output=output_path, seed=seed, ratio=(train_split, val_split, test_split), group_prefix=None, move=False)
    return output_path

'''
Count number of files in 
'''
def countNumberOfFilesInDirectory(dir_path):
    '''
    Count number of files in directory "dir_path"
    
    :param dir_path: Name of directory
    
    :return Count of files in directory
    '''
    return len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])

def getCountOfClasses(train_path, val_path, test_path):
    '''
    Count number of files in "train", "val", "test" sub directories
    
    :param train_path: Name of train directory
    :param val_path: Name of validation directory
    :param test_path: Name of test directory
    
    :return Tuple with number of files in the subdirectories of "train", "val", "test"
    '''
    train_female = countNumberOfFilesInDirectory(train_path + '/Female')
    train_male = countNumberOfFilesInDirectory(train_path + '/Male')
    val_female = countNumberOfFilesInDirectory(val_path + '/Female')
    val_male = countNumberOfFilesInDirectory(val_path + '/Male')
    test_female = countNumberOfFilesInDirectory(test_path + '/Female')
    test_male = countNumberOfFilesInDirectory(test_path + '/Male')
    return (train_female, train_male, val_female, val_male, test_female, test_male)


def readData(folderName, image_size, batch_size, preprocess_input, seed=None, split_folder=True, delete=True, train_split=0.8, val_split=0.1, test_split=0.1):
    '''
    Import train, validation and test data
    
    :param folderName: Name of directory with data
    :param image_size: Preprocess images to size "image_size"
    :param batch_size: Number of data points in batch
    :param batch_size: Number of data points in batch
    :param preprocess_input: Preprocess method for specific neural network
    :param seed: Seed for shuffling and data split
    :param split_folder: If True create new directory with sub directories "train", "val", "test"
    :param delete: If True overwrite folder "folderName_split" 
    :param train_split: Size of train data [0, 1]
    :param val_split: Size of validation data [0, 1]
    :param test_split: Size of test data [0, 1]
    
    :return Tuple with number of files in the subdirectories of "train", "val", "test"
    '''
    
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
    
    count_classes = getCountOfClasses(train_path, val_path, test_path)
    print("Count classes: " + str(count_classes))
    
    return (train_batches, val_batches, test_batches, count_classes)
