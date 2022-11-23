import numpy as np
import tensorflow as tf
from fairness import fairnessMetrics as fm
from model import utils as utils

def build_model(image_size=(224,224), optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), network="Efficient"):

    metrics_list = ["accuracy",
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
        base_model = tf.keras.applications.EfficientNetV2L(include_top=False, input_tensor=inputs, weights=None)
        base_model = utils.addFinalLayersToModel(base_model)
        
    base_model.compile(optimizer=optimizer, 
                               loss="binary_crossentropy", 
                               metrics=metrics_list)
    return base_model