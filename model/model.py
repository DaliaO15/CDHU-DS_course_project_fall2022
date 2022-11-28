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
    

def build_model(image_size=(224,224), network="Efficient"):
    if(network == "VGG"):
        base_model = tf.keras.applications.VGG16(
            weights=None, 
            input_shape=(image_size[0], image_size[1], 3),
            include_top=True,
            classes=1,
            classifier_activation="sigmoid"
        )
        
    elif(network == "Efficient"):
        inputs = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
        base_model = tf.keras.applications.EfficientNetB6(include_top=False, input_tensor=inputs, weights=None)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        base_model = tf.keras.Model(inputs, x)
    
    base_model.build((None, image_size[0], image_size[1], 3))
    
    return base_model
           