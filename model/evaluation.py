from string import Template
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
    
    print("Make folder")
    dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Test"
    os.mkdir(dir_name)
    
    print("Plot Histogram")
    plt.hist(test_predict, bins=100)
    plt.savefig(dir_name + '/histogram.png')
    
    plt.clf()
    
    print("Plot ROC")
    fpr , tpr , thresholds = roc_curve(test_labels, test_predict)
    
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.savefig(dir_name + '/ROC.png')    
    
    plt.clf()
    
    print("Plot Confusion matrix")
    test_predict_labels = np.where(test_predict>0.5, 1, 0)
        
    conf_matrix = confusion_matrix(test_labels, test_predict_labels)
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    plt.savefig(dir_name + '/confusion_matrix.png') 
    
    plt.clf()
    
    print("Plot Results")
    
    tn, fp, fn, tp = conf_matrix.ravel()
    
    labels = ['Female', 'Male']
    true = [tn, tp]
    false = [fn, fp]

    plt.barh(labels, true, color='g')
    plt.barh(labels, false, left=true, color='r')
    plt.savefig(dir_name + '/results.png')
    
    plt.clf()
    
    print(getTemplateText().safe_substitute(tp=tp,fp=fp,tn=tn,fn=fn,
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

    
    
    
def testModelWithThresholdChange():
    return
    
    

    
    
    
def getTemplateText():
    return Template("""
    Test results:
    -------------
    
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