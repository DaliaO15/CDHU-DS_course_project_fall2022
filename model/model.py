import numpy as np
import tensorflow as tf
from fairness import fairnessMetrics as fm
from fairness import customLoss as cl
from model import utils as utils

def metrics_list():
    return ["accuracy",
            tf.keras.metrics.TruePositives(), 
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(), 
            tf.keras.metrics.FalseNegatives(), 
            fm.TruePositiveRate(),
            fm.TrueNegativeRate(),
            fm.FalsePositiveRate(),
            fm.FalseNegativeRate(),
            fm.PositivePredictedValue(),
            fm.FalseDiscoveryRate(),
            fm.NegativePredictedValue(),
            fm.FalseOmissionRate(),
            fm.BinaryDemographicParityDiff(),
            fm.DemographicParity(),
            fm.BinaryEqualizedOddsDiff(),
            fm.BinaryProportionalParityDiff(),
            fm.ProportionalParity(),
            fm.BinaryPredictiveRateParityDiff(),
            fm.PredictiveRateParity(),
            fm.BinaryAccuracyParityDiff(),
            fm.AccuracyParity(),
            fm.BinaryFalseNegativeRateParityDiff(),
            fm.BinaryFalsePositiveRateParityDiff(),
            fm.BinaryNegativePredictiveRateParityDiff(),
            fm.NegativePredictiveRateParity(),
            fm.BinarySpecificityParityDiff()]

def build_model(image_size=(224,224), network="Efficient"):
    if(network == "VGG"):
        base_model = VGGModel(image_size)
        
    elif(network == "Efficient"):
        base_model = EfficientNetModel(image_size)
    
    base_model.build((None, image_size[0], image_size[1], 3))
    
    return base_model

class VGGModel(tf.keras.Model):

    def __init__(self, input_shape, output_shape=None):
        super().__init__()
        self.model = tf.keras.applications.VGG16(
            weights=None, 
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=True,
            classes=1,
            classifier_activation="sigmoid"
        )
        self.verbose = verbose

    @tf.function
    def call(self, inputs, output_shape=None):
        return self.model(inputs)
    
    def keepLayers(self, start, end):
        self.model = tf.keras.Model(self.model.layers[start].input, self.model.layers[end].output)

    def addOutputLayers(self):
        x = tf.keras.layers.GlobalAveragePooling2D()(self.model.layers[-1].output)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(self.model.inputs, x)
    
    def freezeAllLayers(self, freeze=True):
        if(freeze):
            self.model.trainable = False
        else:
            self.model.trainable = True
            
    def freezeCertainLayers(self, end):
        self.model.trainable = True
        for i in range(0,end):
            self.model.layers[i].trainable = False
    
class EfficientNetModel(tf.keras.Model):

    def __init__(self, input_shape, output_shape=None):
        super().__init__()
        inputs = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights=None)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(base_model.inputs, x)

    @tf.function
    def call(self, inputs):
        return self.model(inputs)
    
    def keepLayers(self, start, end):
        self.model = tf.keras.Model(self.model.layers[start].input, self.model.layers[end].output)

    def addOutputLayers(self):
        x = tf.keras.layers.GlobalAveragePooling2D()(self.model.layers[-1].output)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(self.model.inputs, x)
    
    def freezeAllLayers(self, freeze=True):
        if(freeze):
            self.model.trainable = False
        else:
            self.model.trainable = True
            
    def freezeCertainLayers(self, end):
        self.model.trainable = True
        for i in range(0,end):
            self.model.layers[i].trainable = False
            
            
class LoadModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.model = utils.loadModel(dir_name + "/base_model.h5")

    @tf.function
    def call(self, inputs):
        return self.model(inputs)
    
    def keepLayers(self, start, end):
        self.model = tf.keras.Model(self.model.layers[start].input, self.model.layers[end].output)

    def addOutputLayers(self):
        x = tf.keras.layers.GlobalAveragePooling2D()(self.model.layers[-1].output)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(self.model.inputs, x)
    
    def freezeAllLayers(self, freeze=True):
        if(freeze):
            self.model.trainable = False
        else:
            self.model.trainable = True
            
    def freezeCertainLayers(self, end):
        self.model.trainable = True
        for i in range(0,end):
            self.model.layers[i].trainable = False