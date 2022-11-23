import tensorflow as tf

def freezeAllLayers(model):
    model.trainable = False
    
def unfreezeLayers(model, nr):
    for layer in model.layers[nr:]:
        layer.trainable = True
    
def removeLayersEnd(model, nr):
    return tf.keras.Model(model.input, model.layers[:nr].output)