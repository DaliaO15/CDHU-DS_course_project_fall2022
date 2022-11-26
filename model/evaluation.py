from string import Template
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import f1_score     
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import os
import datetime


def testModel(model, ds_test):
    from matplotlib import pyplot as plt
    
    print("Testing Model")
    print("-------------")
    test_predict = model.predict(ds_test, verbose=2)
    test_labels = ds_test.labels
    
    print("Make folder...")
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Test"
    os.mkdir(dir_name)
    
    print("Plot Histogram...")
    plt.hist(test_predict, bins=100)
    plt.savefig(dir_name + '/histogram.png')
    plt.clf()
    
    print("Plot ROC...")
    fpr , tpr , thresholds = roc_curve(test_labels, test_predict)
    
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.savefig(dir_name + '/ROC.png')    
    plt.clf()
    
    print("Plot Confusion matrix...")
    test_predict_labels = np.where(test_predict>0.5, 1, 0)
        
    conf_matrix = confusion_matrix(test_labels, test_predict_labels)
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.savefig(dir_name + '/confusion_matrix.png') 
    plt.clf()
    
    print("Plot Results...")
    
    tn, fp, fn, tp = conf_matrix.ravel()
    
    labels = ['Female', 'Male']
    true = [tn, tp]
    false = [fn, fp]

    plt.barh(labels, true, color='g')
    plt.barh(labels, false, left=true, color='r')
    plt.savefig(dir_name + '/results.png')   
    plt.clf()
    
    print(getTemplateText().safe_substitute(change="Without threshold change", accuracy=(tp+tn)/(tp+fp+tn+fn),
                                           tp=tp,fp=fp,tn=tn,fn=fn,
                                           tpr=tp/(tp+fn), tnr=tn/(tn+fp),
                                           fnr=fn/(tp+fn), fpr=fp/(tn+fp),
                                           ppv=tp/(tp+fp), fdr=fp/(tp+fp),
                                           npv=tn/(tn+fn), fora=fn/(tn+fn),
                                           bdpd=(tp+fp)-(tn+fn), dp=tp+fp,
                                           beod=tp/(tp+fn)-tn/(tn+fp), bppd=(tp+fp)/(tp+fp+tn+fn)-(tn+fn)/(tp+fp+tn+fn),
                                           pp=(tp+fp)/(tp+fp+tn+fn), bprpd=tp/(tp+fp)-tn/(tn+fn),
                                           prp=tp/(tp+fp), bapd=(tp+tn)/(tp+fp+tn+fn)-(fp+fn)/(tp+fp+tn+fn),
                                           ap=(tp+tn)/(tp+fp+tn+fn), bfnrpd=tn/(tn+fp)-tp/(tp+fn),
                                           bfprpd=fp/(tn+fp)-fn/(tp+fn), bnprpd=tn/(tn+fn)-tp/(tp+fp),
                                           nprp=tn/(tn+fn), bspd=tn/(tn+fp)-tp/(tp+fn)))
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
        prop[index]=np.abs((tp/(fp+tf.keras.backend.epsilon()))-(tn/(fn+tf.keras.backend.epsilon())))

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
    
    
def testModelWithThresholdChange(model, ds_val, test_predict, test_labels, dir_name):
    from matplotlib import pyplot as plt
    
    print("Validating Model")
    print("-------------")
    val_predict = model.predict(ds_val, verbose=2)
    val_labels = ds_val.labels
    
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
        plt.clf()
        
        labels = ['Female', 'Male']
        true = [tn, tp]
        false = [fn, fp]

        plt.barh(labels, true, color='g')
        plt.barh(labels, false, left=true, color='r')
        plt.savefig(dir_name + '/threshold_results_' + str_file + '.png')
        plt.clf()
        
        print(getTemplateText().safe_substitute(change=str_results, accuracy=(tp+tn)/(tp+fp+tn+fn),
                                       tp=tp,fp=fp,tn=tn,fn=fn,
                                       tpr=tp/(tp+fn), tnr=tn/(tn+fp),
                                       fnr=fn/(tp+fn), fpr=fp/(tn+fp),
                                       ppv=tp/(tp+fp), fdr=fp/(tp+fp),
                                       npv=tn/(tn+fn), fora=fn/(tn+fn),
                                       bdpd=(tp+fp)-(tn+fn), dp=tp+fp,
                                       beod=tp/(tp+fn)-tn/(tn+fp), bppd=(tp+fp)/(tp+fp+tn+fn)-(tn+fn)/(tp+fp+tn+fn),
                                       pp=(tp+fp)/(tp+fp+tn+fn), bprpd=tp/(tp+fp)-tn/(tn+fn),
                                       prp=tp/(tp+fp), bapd=(tp+tn)/(tp+fp+tn+fn)-(fp+fn)/(tp+fp+tn+fn),
                                       ap=(tp+tn)/(tp+fp+tn+fn), bfnrpd=tn/(tn+fp)-tp/(tp+fn),
                                       bfprpd=fp/(tn+fp)-fn/(tp+fn), bnprpd=tn/(tn+fn)-tp/(tp+fp),
                                       nprp=tn/(tn+fn), bspd=tn/(tn+fp)-tp/(tp+fn)))

    
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