import tensorflow as tf
import datetime
"""
Contribution: Christoph Nötzli
Comments: Erik Norén 14/12-22
"""


def saveModel(model, path):
    model.save(path)
    
def loadModel(path, config):  
    return tf.keras.models.load_model(path, custom_objects=config)

def train_model(model, epochs, ds_train, train_batches, ds_val, val_batches, class_weight=None, weight=True):
    '''
    Train a binary image classification model with training set and validate with validation set. Save training logs.
    
    :param model: model to be trained
    :param epochs: # of epochs for training
    :param ds_train: train data set
    :param train_batches: train batches
    :param ds_val: validation data set
    :param val_batches: batches for validation
    :param class_weight: binary class weights
    :param weight: If True apply class weights to training
    
    
    :return N/A
    ''' 
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    if(weight):
        model.fit(ds_train, steps_per_epoch=(1 + train_batches.n // train_batches.batch_size), callbacks=[tensorboard_callback,earlystop_callback], epochs=epochs, validation_data=ds_val, class_weight=class_weight, validation_steps=(1 + val_batches.n // val_batches.batch_size))
    else:
        model.fit(ds_train, steps_per_epoch=(1 + train_batches.n // train_batches.batch_size), callbacks=[tensorboard_callback,earlystop_callback], epochs=epochs, validation_data=ds_val, validation_steps=(1 + val_batches.n // val_batches.batch_size))
        
def scheduler(epoch, lr):
    """
    Learning rate scheduler. Decreases learning rate depending on current epoch.
    
    :param epoch: current epoch
    :param lr: current learning rate
    
    :return new learning rate
    """
    print('sc')
    if epoch < 5:
        return lr
    elif epoch == 5:
        return lr * tf.math.exp(-0.5)
    elif epoch == 10:
        return lr * tf.math.exp(-0.3)
    elif epoch == 15:
        return lr * tf.math.exp(-0.1)
    else:
        return lr
        
# don't think is being used
def keepLayers(model, start, end):
    return tf.keras.Model(model.layers[start].input, model.layers[end].output)

# don't think is being used
def addOutputLayers(model):
    """
    Connect top layers for binary classification to model
    
    :param model: model to have layers applied to
    
    :return model with applied top layers
    """
    x = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(model.inputs, x)

def freezeAllLayers(model, freeze=True):
    """
    Freezes all layers in a model
    
    :param model: model to have layers frozen
    :param freeze: if True freeze layers
    
    :return N/A
    """
    if(freeze):
        model.trainable = False
    else:
        model.trainable = True

def freezeCertainLayers(model, end):
    """
    Freezes up to a given number of layers in a model.
    
    :param model: model to have layers frozen
    :param end:  the last layer to be frozen
    
    :return N/A
    """
    model.trainable = True
    for i in range(0,end):
        model.layers[i].trainable = False