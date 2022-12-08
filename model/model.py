import numpy as np
import tensorflow as tf
from fairness import fairnessMetrics as fm
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

    
def metrics_dict():
    return {
                'TruePositiveRate': fm.TruePositiveRate(),
                'TrueNegativeRate': fm.TrueNegativeRate(),
                'FalsePositiveRate': fm.FalsePositiveRate(),
                'FalseNegativeRate': fm.FalseNegativeRate(),
                'PositivePredictedValue': fm.PositivePredictedValue(),
                'FalseDiscoveryRate': fm.FalseDiscoveryRate(),
                'NegativePredictedValue': fm.NegativePredictedValue(),
                'FalseOmissionRate': fm.FalseOmissionRate(),
                'BinaryDemographicParityDiff': fm.BinaryDemographicParityDiff(),
                'DemographicParity': fm.DemographicParity(),
                'BinaryEqualizedOddsDiff': fm.BinaryEqualizedOddsDiff(),
                'BinaryProportionalParityDiff': fm.BinaryProportionalParityDiff(),
                'ProportionalParity': fm.ProportionalParity(),
                'BinaryPredictiveRateParityDiff': fm.BinaryPredictiveRateParityDiff(),
                'PredictiveRateParity': fm.PredictiveRateParity(),
                'BinaryAccuracyParityDiff': fm.BinaryAccuracyParityDiff(),
                'AccuracyParity': fm.AccuracyParity(),
                'BinaryFalseNegativeRateParityDiff': fm.BinaryFalseNegativeRateParityDiff(),
                'BinaryFalsePositiveRateParityDiff': fm.BinaryFalsePositiveRateParityDiff(),
                'BinaryNegativePredictiveRateParityDiff': fm.BinaryNegativePredictiveRateParityDiff(),
                'NegativePredictiveRateParity': fm.NegativePredictiveRateParity(),
                'BinarySpecificityParityDiff': fm.BinarySpecificityParityDiff()
    }
    

def build_model(augmentation=False, image_size=(224,224), network="Efficient"):
    if(augmentation):
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.1)
        ])
    if(network == "Xception"):
        inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
                
        # Xception wants inputs rescaled from (0, 255) 
        # to a range of (-1., +1.), the rescaling layer
        # outputs: `(inputs * scale) + offset`
        scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
        inputs = scale_layer(inputs)
        
        if(augmentation):
            x = augmentation_layers(inputs)
            base_model = tf.keras.applications.Xception(weights=None, input_tensor=x, include_top=False)
        else:
            base_model = tf.keras.applications.Xception(weights=None, input_tensor=inputs, include_top=False)
       
        #x = tf.keras.layers.Flatten()(base_model.output)
        #x = tf.keras.layers.Dense(1024, activation='relu')(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        #x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        # This setup for the top layers has not been tested for Xception.
        x = keras.layers.GlobalAveragePooling2D()(base_model.output) 
        x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout (COULD BE REPLACED BY A BATCHNORMALIZATION LAYER
        x = keras.layers.Dense(1, activation="sigmoid")(x)
        
        base_model = tf.keras.Model(inputs, x)
        
        
    elif(network == "Efficient"):
        inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
        if(augmentation):
            x = augmentation_layers(inputs)
            base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=x, drop_connect_rate=0.4)
        else: 
            base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=inputs, drop_connect_rate=0.4)
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        base_model = tf.keras.Model(inputs, x)
        
    return base_model
           