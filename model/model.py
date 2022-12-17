import numpy as np
import tensorflow as tf
from fairness import fairnessMetrics as fm
from model import utils as utils

"""
Contributions: Christoph Nötzli, Erik Norén and Sushruth Badri
Comments: Erik Norén 14/12-22
"""

def metrics_list():
    '''
    :return list of evaluation metrics
    '''
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
    '''
    :return dictionary of evaluation metrics
    '''
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
    '''
    Build image classification model to be used in experiments. Either EfficientNet or Xception.
    
    :param augmentation: If True connect augmentation layers to model
    :param image_size: Size of input image, should be selected according to model specification
    :param network: If "Efficient" EfficientNet is selected as model, 
                    If "Xception" Xception is selected as model
    
    :return keras model
    '''
    if(augmentation):
        # create augmentation layers
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomContrast(0.1)
        ])
    if(network == "Xception"):
        inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
                
        if(augmentation):
            x = augmentation_layers(inputs)
            # Rescaling layer with appropriate preprocessing for Xception
            x = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1)(x)
            base_model = tf.keras.applications.Xception(weights=None, input_tensor=x, include_top=False)
        else:
            # Rescaling layer with appropriate preprocessing for Xception
            x = tf.keras.layers.Rescaling(scale=1.0/127.5, offset=-1)(inputs)
            base_model = tf.keras.applications.Xception(weights=None, input_tensor=inputs, include_top=False)
        
        # Apply top layers for binary classification
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        
        base_model = tf.keras.Model(inputs, x)
        
        
    elif(network == "Efficient"):
        inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
        if(augmentation):
            x = augmentation_layers(inputs)
            base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=x, drop_connect_rate=0.4)
        else: 
            base_model = tf.keras.applications.EfficientNetB3(include_top=False, input_tensor=inputs, drop_connect_rate=0.4)
        
        # Apply top layers for binary classification
        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        base_model = tf.keras.Model(inputs, x)
        
    return base_model
           