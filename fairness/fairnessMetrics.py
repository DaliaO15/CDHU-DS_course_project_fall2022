from tensorflow import keras
from keras import backend

class TruePositiveRate(keras.metrics.Metric):
    def __init__(self, name='true_positive_rate', **kwargs):
        super(TruePositiveRate, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fn = keras.metrics.FalseNegatives()
        self.tpr = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.tpr = self.tp.result()/(self.tp.result() +  self.fn.result() + backend.epsilon())
        
    def result(self):
        return self.tpr

    def reset_state(self):
        self.tp.reset_state()
        self.fn.reset_state()
        self.tpr = 0.0
        
        
class TrueNegativeRate(keras.metrics.Metric):
    def __init__(self, name='true_negative_rate', **kwargs):
        super(TrueNegativeRate, self).__init__(name=name, **kwargs)
        self.tn = keras.metrics.TrueNegatives()
        self.fp = keras.metrics.FalsePositives()
        self.tnr = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tnr = self.tn.result()/(self.tn.result() +  self.fp.result() + backend.epsilon())
        
    def result(self):
        return self.tnr

    def reset_state(self):
        self.tn.reset_state()
        self.fp.reset_state()
        self.tnr = 0.0   

        
class FalseNegativeRate(keras.metrics.Metric):
    def __init__(self, name='false_negative_rate', **kwargs):
        super(FalseNegativeRate, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fn = keras.metrics.FalseNegatives()
        self.fnr = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.fnr = self.fn.result()/(self.tp.result() +  self.fn.result() + backend.epsilon())
        
    def result(self):
        return self.fnr

    def reset_state(self):
        self.tp.reset_state()
        self.fn.reset_state()
        self.fnr = 0.0  
        
        
class FalsePositiveRate(keras.metrics.Metric):
    def __init__(self, name='false_positive_rate', **kwargs):
        super(FalsePositiveRate, self).__init__(name=name, **kwargs)
        self.tn = keras.metrics.TrueNegatives()
        self.fp = keras.metrics.FalsePositives()
        self.fpr = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.fpr = self.fp.result()/(self.tn.result() +  self.fp.result() + backend.epsilon())
        
    def result(self):
        return self.fpr

    def reset_state(self):
        self.tn.reset_state()
        self.fp.reset_state()
        self.fpr = 0.0
        
        
class PositivePredictedValue(keras.metrics.Metric):
    def __init__(self, name='positive_predicted_value', **kwargs):
        super(PositivePredictedValue, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.ppv = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.ppv = self.tp.result()/(self.tp.result() +  self.fp.result() + backend.epsilon())
                                     
    def result(self):
        return self.ppv

    def reset_state(self):
        self.tp.reset_state()
        self.fp.reset_state()
        self.ppv = 0.0
        

class FalseDiscoveryRate(keras.metrics.Metric):
    def __init__(self, name='false_discovery_rate', **kwargs):
        super(FalseDiscoveryRate, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.fdr = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight) 
        self.fdr = self.fp.result()/(self.tp.result() +  self.fp.result() + backend.epsilon())
        
    def result(self):
        return self.fdr

    def reset_state(self):
        self.tp.reset_state()
        self.fp.reset_state()
        self.fdr = 0.0
        
        
class NegativePredictedValue(keras.metrics.Metric):
    def __init__(self, name='negative_predicted_value', **kwargs):
        super(NegativePredictedValue, self).__init__(name=name, **kwargs)
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.npv = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.npv = self.tn.result()/(self.tn.result() +  self.fn.result() + backend.epsilon())
        
    def result(self):
        return self.npv

    def reset_state(self):
        self.tn.reset_state()
        self.fn.reset_state()
        self.npv = 0.0

        
class FalseOmissionRate(keras.metrics.Metric):
    def __init__(self, name='false_omission_rate', **kwargs):
        super(FalseOmissionRate, self).__init__(name=name, **kwargs)
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.fora = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.fora = self.fn.result()/(self.tn.result() +  self.fn.result() + backend.epsilon())
        
    def result(self):
        return self.fora

    def reset_state(self):
        self.tn.reset_state()
        self.fn.reset_state()
        self.fora = 0.0
        

class BinaryDemographicParityDiff(keras.metrics.Metric):
    def __init__(self, name='binary_demographic_parity_diff', **kwargs):
        super(BinaryDemographicParityDiff, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.bdp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.bdp = (self.tp.result() + self.fp.result())-(self.tn.result() +  self.fn.result() + backend.epsilon())
        
    def result(self):
        return self.bdp

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.bdp = 0.0

        
class DemographicParity(keras.metrics.Metric):
    def __init__(self, name='demographic_parity', **kwargs):
        super(DemographicParity, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.dp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.dp = (self.tp.result() + self.fp.result() + backend.epsilon())
        
    def result(self):
        return self.dp

    def reset_state(self):
        self.tp.reset_state()
        self.fp.reset_state()
        self.dp = 0.0
   

class BinaryEqualizedOddsDiff(keras.metrics.Metric):
    def __init__(self, name='binary_equalized_odds_diff', **kwargs):
        super(BinaryEqualizedOddsDiff, self).__init__(name=name, **kwargs)
        self.tpr1 = TruePositiveRate()
        self.tpr2 = TruePositiveRate()
        self.beop = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tpr1.update_state(y_true, y_pred, sample_weight)
        self.tpr2.update_state(1-y_true, 1-y_pred, sample_weight)
        self.beop = self.tpr2.result()-self.tpr1.result()
        
    def result(self):
        return self.beop

    def reset_state(self):
        self.tpr1.reset_state()
        self.tpr2.reset_state()
        self.beop = 0.0
        
    
class BinaryProportionalParityDiff(keras.metrics.Metric):
    def __init__(self, name='binary_proportional_parity_diff', **kwargs):
        super(BinaryProportionalParityDiff, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.bpp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.bpp = ((self.tp.result() + self.fp.result())/(self.tp.result() + self.fp.result() + self.tn.result() + self.fn.result() + backend.epsilon()))-((self.tn.result() +  self.fn.result())/(self.tp.result() + self.fp.result() + self.tn.result() + self.fn.result()))
        
    def result(self):
        return self.bpp

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.bpp = 0.0


class ProportionalParity(keras.metrics.Metric):
    def __init__(self, name='proportional_parity', **kwargs):
        super(ProportionalParity, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.pp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.pp = ((self.tp.result() + self.fp.result())/(self.tp.result() + self.fp.result() + self.tn.result() + self.fn.result() + backend.epsilon()))
        
    def result(self):
        return self.pp

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.pp = 0.0        
    
        
class BinaryPredictiveRateParityDiff(keras.metrics.Metric):
    def __init__(self, name='binary_predictive_rate_parity_diff', **kwargs):
        super(BinaryPredictiveRateParityDiff, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.bprp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.bprp = (self.tp.result()/(self.tp.result() + self.fp.result()  + backend.epsilon()))-(self.tn.result()/(self.tn.result() + self.fn.result() + backend.epsilon()))
        
    def result(self):
        return self.bprp

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.bprp = 0.0
        
        
class PredictiveRateParity(keras.metrics.Metric):
    def __init__(self, name='predictive_rate_parity', **kwargs):
        super(PredictiveRateParity, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.prp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.prp = (self.tp.result()/(self.tp.result() + self.fp.result()  + backend.epsilon()))
        
    def result(self):
        return self.prp

    def reset_state(self):
        self.tp.reset_state()
        self.fp.reset_state()
        self.prp = 0.0
        
        
class BinaryAccuracyParityDiff(keras.metrics.Metric):
    def __init__(self, name='binary_accuracy_parity_diff', **kwargs):
        super(BinaryAccuracyParityDiff, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.bap = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.bap = ((self.tp.result() + self.tn.result())/(self.tp.result() + self.fp.result() + self.tn.result() + self.fn.result() + backend.epsilon()))-((self.fp.result() +  self.fn.result())/(self.tp.result() + self.fp.result() + self.tn.result() + self.fn.result()))
        
    def result(self):
        return self.bap

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.bap = 0.0


class AccuracyParity(keras.metrics.Metric):
    def __init__(self, name='accuracy_parity', **kwargs):
        super(AccuracyParity, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.ap = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.ap = ((self.tp.result() + self.tn.result())/(self.tp.result() + self.fp.result() + self.tn.result() + self.fn.result() + backend.epsilon()))
        
    def result(self):
        return self.ap

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.ap = 0.0        
        
        
class BinaryFalseNegativeRateParityDiff(keras.metrics.Metric):
    def __init__(self, name='false_negative_rate_parity_diff', **kwargs):
        super(BinaryFalseNegativeRateParityDiff, self).__init__(name=name, **kwargs)
        self.fnr1 = FalseNegativeRate()
        self.fnr2 = FalseNegativeRate()
        self.fnrp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.fnr1.update_state(y_true, y_pred, sample_weight)
        self.fnr2.update_state(1-y_true, 1-y_pred, sample_weight)
        self.fnrp = self.fnr2.result()-self.fnr1.result()
        
    def result(self):
        return self.fnrp

    def reset_state(self):
        self.fnr1.reset_state()
        self.fnr2.reset_state()
        self.fnrp = 0.0

        
class BinaryFalsePositiveRateParityDiff(keras.metrics.Metric):
    def __init__(self, name='false_positive_rate_parity_diff', **kwargs):
        super(BinaryFalsePositiveRateParityDiff, self).__init__(name=name, **kwargs)
        self.fpr1 = FalsePositiveRate()
        self.fpr2 = FalsePositiveRate()
        self.fprp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.fpr1.update_state(y_true, y_pred, sample_weight)
        self.fpr2.update_state(1-y_true, 1-y_pred, sample_weight)
        self.fprp = self.fpr2.result()-self.fpr1.result()
        
    def result(self):
        return self.fprp

    def reset_state(self):
        self.fpr1.reset_state()
        self.fpr2.reset_state()
        self.fprp = 0.0


class BinaryNegativePredictiveRateParityDiff(keras.metrics.Metric):
    def __init__(self, name='binary_negative_predictive_rate_parity_diff', **kwargs):
        super(BinaryNegativePredictiveRateParityDiff, self).__init__(name=name, **kwargs)
        self.tp = keras.metrics.TruePositives()
        self.fp = keras.metrics.FalsePositives()
        self.tn = keras.metrics.TrueNegatives()
        self.fn = keras.metrics.FalseNegatives()
        self.bnprp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tp.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.bnprp = (self.tn.result()/(self.tn.result() + self.fn.result()  + backend.epsilon()))-(self.tp.result()/(self.tp.result() + self.fp.result() + backend.epsilon()))
        
    def result(self):
        return self.bnprp

    def reset_state(self):
        self.tp.reset_state()
        self.tn.reset_state()
        self.fp.reset_state()
        self.fn.reset_state()
        self.bnprp = 0.0
        
        
class NegativePredictiveRateParity(keras.metrics.Metric):
    def __init__(self, name='negative_predictive_rate_parity', **kwargs):
        super(NegativePredictiveRateParity, self).__init__(name=name, **kwargs)
        self.tn = keras.metrics.TruePositives()
        self.fn = keras.metrics.FalsePositives()
        self.nprp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fn.update_state(y_true, y_pred, sample_weight)
        self.nprp = (self.tn.result()/(self.tn.result() + self.fn.result()  + backend.epsilon()))
        
    def result(self):
        return self.nprp

    def reset_state(self):
        self.tn.reset_state()
        self.fn.reset_state()
        self.nprp = 0.0
        
class BinarySpecificityParityDiff(keras.metrics.Metric):
    def __init__(self, name='binary_specificity_parity_diff', **kwargs):
        super(BinarySpecificityParityDiff, self).__init__(name=name, **kwargs)
        self.tnr1 = FalseNegativeRate()
        self.tnr2 = FalseNegativeRate()
        self.bsp = 0.0
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tnr1.update_state(y_true, y_pred, sample_weight)
        self.tnr2.update_state(1-y_true, 1-y_pred, sample_weight)
        self.bsp = self.tnr2.result()-self.tnr1.result()
        
    def result(self):
        return self.bsp

    def reset_state(self):
        self.tnr1.reset_state()
        self.tnr2.reset_state()
        self.bsp = 0.0
