import tensorflow as tf
import datetime

def saveModel(model, path):
    model.save(path)
    
def loadModel(path):  
    return tf.keras.models.load_model(path,compile=False)
    
def train_model(model, epochs, train_ds, val_ds, class_weight=None, weight=True):
    schedulercb = tf.keras.callbacks.LearningRateScheduler(scheduler)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if(weight):
        model.fit(train_ds, callbacks=[schedulercb,tensorboard_callback], epochs=epochs, validation_data=val_ds, class_weight=class_weight, verbose=2)
    else:
        model.fit(train_ds, callbacks=[schedulercb,tensorboard_callback], epochs=epochs, validation_data=val_ds, verbose=2)
        
def scheduler(epoch, lr):
    print('sc')
    if epoch < 10:
        return lr
    elif epoch == 10:
        return lr * tf.math.exp(-0.5)
    elif epoch == 30:
        return lr * tf.math.exp(-0.3)
    elif epoch == 50:
        return lr * tf.math.exp(-0.1)
    else:
        return lr
    
def addFinalLayersToModel(model, image_size=(224, 224)):    
    x = tf.keras.layers.GlobalAveragePooling2D()(model.layers[-1].output)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    return tf.keras.Model(inputs=model.inputs, outputs=x)