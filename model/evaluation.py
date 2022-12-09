from string import Template
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import f1_score     
from sklearn.metrics import plot_confusion_matrix
import data.imageReading as ir
from model import model as m
from model import utils as utils
import numpy as np
import os
import datetime


def testModel(model, ds_test, test_batches, dir_name):
    from matplotlib import pyplot as plt
    
    print("Testing Model")
    print("-------------")
    test_predict = model.predict(ds_test, steps=(1 + test_batches.n // test_batches.batch_size))
    test_labels = test_batches.labels
    
    print("Plot Histogram...")
    plt.hist(test_predict, bins=100)
    plt.savefig(dir_name + '/histogram.png')
    plt.show()
    plt.clf()
    
    print("Plot ROC...")
    fpr , tpr , thresholds = roc_curve(test_labels, test_predict)
    
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.savefig(dir_name + '/ROC.png')    
    plt.show()
    plt.clf()
    
    print("Plot Confusion matrix...")
    test_predict_labels = np.where(test_predict>0.5, 1, 0)
        
    conf_matrix = confusion_matrix(test_labels, test_predict_labels)
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.savefig(dir_name + '/confusion_matrix.png') 
    plt.show()
    plt.clf()
    
    print("Plot Results...")
    
    tn, fp, fn, tp = conf_matrix.ravel()
    
    labels = ['Female', 'Male']
    true = [tn, tp]
    false = [fn, fp]

    plt.barh(labels, true, color='g')
    plt.barh(labels, false, left=true, color='r')
    plt.savefig(dir_name + '/results.png')   
    plt.show()
    plt.clf()
    
    print(getTemplateText().safe_substitute(change="Without threshold change", 
                                           accuracy=format((tp+tn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           tp=format(tp,".3f"),fp=format(fp,".3f"),tn=format(tn,".3f"),fn=format(fn,".3f"),
                                           tpr=format(tp/(tp+fn+tf.keras.backend.epsilon()),".3f"), 
                                           tnr= format(tn/(tn+fp+tf.keras.backend.epsilon()),".3f"),
                                           fnr=format(fn/(tp+fn+tf.keras.backend.epsilon()),".3f"), 
                                           fpr=format(fp/(tn+fp+tf.keras.backend.epsilon()),".3f"),
                                           ppv=format(tp/(tp+fp+tf.keras.backend.epsilon()),".3f"), 
                                           fdr=format(fp/(tp+fp+tf.keras.backend.epsilon()),".3f"),
                                           npv=format(tn/(tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           fora=format(fn/(tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           bdpd=format((tp+fp)-(tn+fn),".3f"), dp=format(tp+fp,".3f"),
                                           beod=format(tp/(tp+fn+tf.keras.backend.epsilon())-tn/(tn+fp+tf.keras.backend.epsilon()),".3f"), 
                                           bppd=format((tp+fp)/(tp+fp+tn+fn+tf.keras.backend.epsilon())-(tn+fn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           pp=format((tp+fp)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bprpd=format(tp/(tp+fp+tf.keras.backend.epsilon())-tn/(tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           prp=format(tp/(tp+fp+tf.keras.backend.epsilon()),".3f"), 
                                           bapd=format((tp+tn)/(tp+fp+tn+fn+tf.keras.backend.epsilon())-(fp+fn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           ap=format((tp+tn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bfnrpd=format(tn/(tn+fp)-tp/(tp+fn+tf.keras.backend.epsilon()),".3f"),
                                           bfprpd=format(fp/(tn+fp+tf.keras.backend.epsilon())-fn/(tp+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bnprpd=format(tn/(tn+fn+tf.keras.backend.epsilon())-tp/(tp+fp+tf.keras.backend.epsilon()),".3f"),
                                           nprp=format(tn/(tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bspd=format(tn/(tn+fp+tf.keras.backend.epsilon())-tp/(tp+fn+tf.keras.backend.epsilon()),".3f")))
    return (test_predict, test_labels, dir_name)
    
def findIdealThreshold(model, val_predict, val_labels):
    thresholds = np.arange(0.0, 1.0, 0.0001)
    true = np.zeros(shape=(len(thresholds)))
    false = np.zeros(shape=(len(thresholds)))
    total = np.zeros(shape=(len(thresholds)))
    prop = np.zeros(shape=(len(thresholds)))

    for index, elem in enumerate(thresholds):
        val_predicted_labels = np.where(val_predict>elem, 1, 0)
        tn, fp, fn, tp = confusion_matrix(val_labels, val_predicted_labels).ravel()
        true[index]=np.abs(tp-tn)
        false[index]=np.abs(fp-fn)
        total[index]=np.abs((tp+fp)-(tn+fn))
        prop[index]=np.abs((tp/(tp+fn+tf.keras.backend.epsilon()))-(tn/(tn+fp+tf.keras.backend.epsilon())))

    index_true = np.argmin(true)
    index_false = np.argmin(false)
    index_total = np.argmin(total)
    index_prop = np.argmin(prop)
    thresholdOpt_true = round(thresholds[index_true], ndigits = 4)
    thresholdOpt_false = round(thresholds[index_false], ndigits = 4)
    thresholdOpt_total = round(thresholds[index_total], ndigits = 4)
    thresholdOpt_prop = round(thresholds[index_prop], ndigits = 4)
    Opt_true = round(true[index_true], ndigits = 4)
    Opt_false = round(true[index_false], ndigits = 4)
    Opt_total = round(true[index_total], ndigits = 4)
    Opt_prop = round(true[index_prop], ndigits = 4)
    print()
    print('Thresholds: ')
    print('-----------')
    print('Best Equal True Threshold: {} with Opt: {}'.format(thresholdOpt_true, Opt_true))
    print('Best Equal False Threshold: {} with Opt: {}'.format(thresholdOpt_false, Opt_false))
    print('Best Equal Predicted Threshold: {} with Opt: {}'.format(thresholdOpt_total, Opt_total))
    print('Best Equal Odds Threshold: {} with Opt: {}'.format(thresholdOpt_prop, Opt_prop))
    
    return ((thresholdOpt_true, thresholdOpt_false, thresholdOpt_total, thresholdOpt_prop),
            (true, false, total, prop),
            thresholds)
    
    
def testModelWithThresholdChange(model, ds_val, val_batches, test_predict, test_labels, dir_name):
    from matplotlib import pyplot as plt
    
    print("Validating Model")
    print("-------------")
    val_predict = model.predict(ds_val, steps=(1 + val_batches.n // val_batches.batch_size), verbose=2)
    val_labels = val_batches.labels
    
    print("Calcualte Optimal Thresholds...")
    thresholds, values, threshold_values = findIdealThreshold(model, val_predict, val_labels)
    
    print("Plot results with threshold change...")
    for index, threshold in enumerate(thresholds):
        test_predicted_labels = np.where(test_predict>threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(test_labels, test_predicted_labels).ravel()
    
        if(index == 0):
            str_file = 'equal_true'
            str_results = 'Equal True'
        elif(index == 1):
            str_file = 'equal_false'
            str_results = 'Equal False'
        elif(index == 2):
            str_file = 'equal_total'
            str_results = 'Equal Total'    
        elif(index == 3):
            str_file = 'equal_odds'
            str_results = 'Equal Odds'  
            
        plt.plot(threshold_values[500:9500], values[index][500:9500]) 
        plt.savefig(dir_name + '/threshold_' + str_file + '.png')
        plt.show()
        plt.clf()
        
        labels = ['Female', 'Male']
        true = [tn, tp]
        false = [fn, fp]

        plt.barh(labels, true, color='g')
        plt.barh(labels, false, left=true, color='r')
        plt.savefig(dir_name + '/threshold_results_' + str_file + '.png')
        plt.show()
        plt.clf()
        
        print(getTemplateText().safe_substitute(change=str_results, 
                                           accuracy=format((tp+tn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           tp=format(tp,".3f"),fp=format(fp,".3f"),tn=format(tn,".3f"),fn=format(fn,".3f"),
                                           tpr=format(tp/(tp+fn+tf.keras.backend.epsilon()),".3f"), 
                                           tnr= format(tn/(tn+fp+tf.keras.backend.epsilon()),".3f"),
                                           fnr=format(fn/(tp+fn+tf.keras.backend.epsilon()),".3f"), 
                                           fpr=format(fp/(tn+fp+tf.keras.backend.epsilon()),".3f"),
                                           ppv=format(tp/(tp+fp+tf.keras.backend.epsilon()),".3f"), 
                                           fdr=format(fp/(tp+fp+tf.keras.backend.epsilon()),".3f"),
                                           npv=format(tn/(tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           fora=format(fn/(tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           bdpd=format((tp+fp)-(tn+fn),".3f"), dp=format(tp+fp,".3f"),
                                           beod=format(tp/(tp+fn+tf.keras.backend.epsilon())-tn/(tn+fp+tf.keras.backend.epsilon()),".3f"), 
                                           bppd=format((tp+fp)/(tp+fp+tn+fn+tf.keras.backend.epsilon())-(tn+fn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           pp=format((tp+fp)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bprpd=format(tp/(tp+fp+tf.keras.backend.epsilon())-tn/(tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           prp=format(tp/(tp+fp+tf.keras.backend.epsilon()),".3f"), 
                                           bapd=format((tp+tn)/(tp+fp+tn+fn+tf.keras.backend.epsilon())-(fp+fn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"),
                                           ap=format((tp+tn)/(tp+fp+tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bfnrpd=format(tn/(tn+fp)-tp/(tp+fn+tf.keras.backend.epsilon()),".3f"),
                                           bfprpd=format(fp/(tn+fp+tf.keras.backend.epsilon())-fn/(tp+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bnprpd=format(tn/(tn+fn+tf.keras.backend.epsilon())-tp/(tp+fp+tf.keras.backend.epsilon()),".3f"),
                                           nprp=format(tn/(tn+fn+tf.keras.backend.epsilon()),".3f"), 
                                           bspd=format(tn/(tn+fp+tf.keras.backend.epsilon())-tp/(tp+fn+tf.keras.backend.epsilon()),".3f")))
        
    
def getTemplateText():
    return Template("""
    Test results ($change):
    -------------
    
    Accuracy: $accuracy
    
    True positives: $tp
    False positives: $fp
    
    True negatives: $tn
    False negatives: $fn
    
    Test results metrics:
    ---------------------
    True positive rate tp/(tp+fn): $tpr
    True negative rate tn/(tn+fp): $tnr
    
    False negative rate fn/(tp+fn): $fnr
    False positive rate fp/(tn+fp): $fpr
    
    Positive predicted value tp/(tp+fp): $ppv
    False discovery rate fp/(tp+fp): $fdr
    
    Negative predicted value tn/(tn+fn): $npv
    False omission rate fn/(tn+fn): $fora
    
    
    Binary demographic parity diff (tp+fp)-(tn+fn): $bdpd
    Demographic parity tp+fp: $dp
    
    Binary equalized odds diff (tp/(tp+fn))-(tn/(tn+fp)): $beod
    
    Binary proportional parity diff ((tp+fp)/(tp+fp+tn+fn))-((tn+fn)/(tp+fp+tn+fn)): $bppd
    Proportional parity (tp+fp)/(tp+fp+tn+fn): $pp
    
    Binary predictive rate parity diff (tp/(tp+fp))-(tn/(tn+fn)): $bprpd
    Predictive rate parity tp/(tp+fp): $prp
    
    Binary accuracy parity diff (tp+tn)/(tp+fp+tn+fn)-(fp+fn)/(tp+fp+tn+fn): $bapd
    Accuracy parity (tp+tn)/(tp+fp+tn+fn): $ap
    
    Binary false negative rate parity diff tn/(tn+fp)-tp/(tp+fn): $bfnrpd
    Binary false positive rate parity diff fp/(tn+fp)-fn/(tp+fn): $bfprpd
    
    Binary negative predictive rate parity diff tn/(tn+fn)-(tp/(tp+fp): $bnprpd
    
    Negative predictive rate parity tn/(tn+fn): $nprp
    
    Binary specificity parity diff tn/(tn+fp)-tp/(tp+fn): $bspd
    """)