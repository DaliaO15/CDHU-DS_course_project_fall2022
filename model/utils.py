import tensorflow as tf
import datetime

def saveModel(model, path):
    model.save(path)
    
def loadModel(path):  
    return tf.keras.models.load_model(path,compile=False)
    
def train_model(model, epochs, ds_train, train_batches, ds_val, class_weight=None, weight=True):
    schedulercb = tf.keras.callbacks.LearningRateScheduler(scheduler)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if(weight):
        model.fit(ds_train, steps_per_epoch=(1 + train_batches.n // train_batches.batch_size), callbacks=[schedulercb,tensorboard_callback], epochs=epochs, validation_data=ds_val, class_weight=class_weight)
    else:
        model.fit(ds_train, steps_per_epoch=(1 + train_batches.n // train_batches.batch_size), callbacks=[schedulercb,tensorboard_callback], epochs=epochs, validation_data=ds_val)
        
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