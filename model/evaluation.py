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
    test_predict = model.predict(ds_test, steps=(1 + test_batches.n // test_batches.batch_size), verbose=2)
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


def kfoldCrossValidation(data_directory, kfoldmodel, preprocess, image_size=(224,224), nr_of_splits=10, epochs=30, class_weight=False):
    ds_path = ir.createFolders(data_directory, 1, 0, 0, True, None)

    ds_train = tf.keras.utils.image_dataset_from_directory(
        ds_path + "/train",
        label_mode = "binary",
        image_size=(image_size[0], image_size[1])
    )
    
    preprocess(ds_train)
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds_train = ds_train.with_options(options)
    ds_test = ds_test.with_options(options)

    train_splits = []
    for i in range(0,nr_of_splits):
        train_splits.append(ds_train.shard(num_shards=nr_of_splits, index=i))
    

    history = []
    for i in range(0,nr_of_splits):
        ds_val_fold = train_splits.pop(0)
        ds_train_fold = train_splits[0]
        for j in range(1, nr_of_splits-1):
            ds_train_fold = ds_train_fold.concatenate(train_splits[j])

        
        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            model = tf.keras.models.clone_model(kfoldmodel)
            metrics_list = m.metrics_list()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                                   loss="binary_crossentropy", metrics=metrics_list)

        schedulercb = tf.keras.callbacks.LearningRateScheduler(utils.scheduler)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, baseline=0.7)
        
        if(class_weight):
            class_weights = mit.findClassWeights(ds_train_fold)
        else: 
            class_weights = None
        
        test_results = {}
        
        test_results = model.fit(ds_train_fold, class_weight=class_weight, callbacks=[schedulercb,tensorboard_callback,earlystop_callback], epochs=epochs, validation_data=ds_val_fold, verbose=2)

        history.append(test_results.history)    
        train_splits.append(ds_val_fold)    
        
    accuracy = [element[-1] for element in history["accuray"]]
    loss = [element[-1] for element in history["loss"]]

    tpr = [element[-1] for element in history["true_positive_rate"]]
    tnr = [element[-1] for element in history["true_negative_rate"]]
    fpr = [element[-1] for element in history["false_positive_rate"]]
    fnr = [element[-1] for element in history["false_negative_rate"]]
    
    ppv = [element[-1] for element in history["positive_predicted_value"]]
    fdr = [element[-1] for element in history["false_discovery_rate"]]
    npv = [element[-1] for element in history["negative_predicted_value"]]
    fora = [element[-1] for element in history["false_omission_rate"]]
    
    bdpd = [element[-1] for element in history["binary_demographic_parity_diff"]]
    beod = [element[-1] for element in history["binary_equalized_odds_diff"]]
    bspd = [element[-1] for element in history["binary_specificity_parity_diff"]]
    
    
    vaccuracy = [element[-1] for element in history["val_accuray"]]
    vloss = [element[-1] for element in history["val_loss"]]
    
    
    vtpr = [element[-1] for element in history["val_true_positive_rate"]]
    vtnr = [element[-1] for element in history["val_true_negative_rate"]]
    vfpr = [element[-1] for element in history["val_false_positive_rate"]]
    vfnr = [element[-1] for element in history["val_false_negative_rate"]]
    
    vppv = [element[-1] for element in history["val_positive_predicted_value"]]
    vfdr = [element[-1] for element in history["val_false_discovery_rate"]]
    vnpv = [element[-1] for element in history["val_negative_predicted_value"]]
    vfora = [element[-1] for element in history["val_false_omission_rate"]]
    
    vbdpd = [element[-1] for element in history["val_binary_demographic_parity_diff"]]
    vbeod = [element[-1] for element in history["val_binary_equalized_odds_diff"]]
    vbspd = [element[-1] for element in history["val_binary_specificity_parity_diff"]]
    
    
    
    print(getTemplateCrossValidation().safe_substitute( 
       accuracy=format(np.mean(accuracy),".3f") + "+/-" + format(np.std(accuracy),".3f"),
       loss=format(np.mean(loss),".3f") + "+/-" + format(np.std(loss),".3f"),
       tpr=format(np.mean(tpr),".3f") + "+/-" + format(np.std(tpr),".3f"),
       tnr=format(np.mean(tnr),".3f") + "+/-" + format(np.std(tnr),".3f"),
       fpr=format(np.mean(fpr),".3f") + "+/-" + format(np.std(fpr),".3f"),
       fnr=format(np.mean(fnr),".3f") + "+/-" + format(np.std(fnr),".3f"),
       ppv=format(np.mean(ppv),".3f") + "+/-" + format(np.std(ppv),".3f"),
       fdr=format(np.mean(fdr),".3f") + "+/-" + format(np.std(fdr),".3f"),
       npv=format(np.mean(npv),".3f") + "+/-" + format(np.std(npv),".3f"),
       fora=format(np.mean(fora),".3f") + "+/-" + format(np.std(fora),".3f"),
       bdpd=format(np.mean(bdpd),".3f") + "+/-" + format(np.std(bdpd),".3f"),
       beod=format(np.mean(beod),".3f") + "+/-" + format(np.std(beod),".3f"),
       bspd=format(np.mean(bspd),".3f") + "+/-" + format(np.std(bspd),".3f"),
       vaccuracy=format(np.mean(vaccuracy),".3f") + "+/-" + format(np.std(vaccuracy),".3f"),
       vloss=format(np.mean(vloss),".3f") + "+/-" + format(np.std(vloss),".3f"),
       vtpr=format(np.mean(vtpr),".3f") + "+/-" + format(np.std(vtpr),".3f"),
       vtnr=format(np.mean(vtnr),".3f") + "+/-" + format(np.std(vtnr),".3f"),
       vfpr=format(np.mean(vfpr),".3f") + "+/-" + format(np.std(vfpr),".3f"),
       vfnr=format(np.mean(vfnr),".3f") + "+/-" + format(np.std(vfnr),".3f"),
       vppv=format(np.mean(vppv),".3f") + "+/-" + format(np.std(vppv),".3f"),
       vfdr=format(np.mean(vfdr),".3f") + "+/-" + format(np.std(vfdr),".3f"),
       vnpv=format(np.mean(vnpv),".3f") + "+/-" + format(np.std(vnpv),".3f"),
       vfora=format(np.mean(vfora),".3f") + "+/-" + format(np.std(vfora),".3f"),
       vbdpd=format(np.mean(vbdpd),".3f") + "+/-" + format(np.std(vbdpd),".3f"),
       vbeod=format(np.mean(vbeod),".3f") + "+/-" + format(np.std(vbeod),".3f"),
       vbspd=format(np.mean(vbspd),".3f") + "+/-" + format(np.std(vbspd),".3f")
    ))
    return history
        
        
    
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
    
    
def getTemplateCrossValidation():
    return Template("""
    Cross Validation results:
    -------------
    
    Accuracy: $accuracy
    Loss: $loss
    
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
    
    Binary equalized odds diff (tp/(tp+fn))-(tn/(tn+fp)): $beod

    
    Binary specificity parity diff tn/(tn+fp)-tp/(tp+fn): $bspd
    
    
    Cross Validation validation results:
    -------------
    
    Accuracy: $vaccuracy
    Loss: $vloss
    
    Test results metrics:
    ---------------------
    True positive rate tp/(tp+fn): $vtpr
    True negative rate tn/(tn+fp): $vtnr
    
    False negative rate fn/(tp+fn): $vfnr
    False positive rate fp/(tn+fp): $vfpr
    
    Positive predicted value tp/(tp+fp): $vppv
    False discovery rate fp/(tp+fp): $vfdr
    
    Negative predicted value tn/(tn+fn): $vnpv
    False omission rate fn/(tn+fn): $vfora
    
    Binary demographic parity diff (tp+fp)-(tn+fn): $vbdpd
    
    Binary equalized odds diff (tp/(tp+fn))-(tn/(tn+fp)): $vbeod

    Binary specificity parity diff tn/(tn+fp)-tp/(tp+fn): $vbspd
    """)