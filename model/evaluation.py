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
from sklearn.model_selection import KFold
import numpy as np
import os
import datetime
import glob
import random

"""
Contribution: Christoph Nötzli
Comments: Christoph Nötzli, Erik Norén 14/12-22
"""

def testModel(model, ds_test, test_batches, dir_name):
    '''
    Test a image classification model on a test set. Prints and saves results in given directory.
    
    :param model: trained model to be tested
    :param ds_test: test data set
    :param test_batches: test batches
    :param dir_name: name of directory where test results should be saved
    
    :return Tuple with test predictions, test labels, and dir_name
    '''
    
    from matplotlib import pyplot as plt
    
    print("Testing Model")
    print("-------------")
    test_predict = model.predict(ds_test, steps=(1 + test_batches.n // test_batches.batch_size))
    test_labels = test_batches.labels
    
    print("Plot Histogram...")
    plt.title('Histogram Predicted Values')
    plt.hist(test_predict, bins=100)
    plt.xlabel('predicted value') 
    plt.savefig(dir_name + '/histogram.png')
    plt.show()
    plt.clf()
    
    print("Plot ROC...")
    fpr , tpr , thresholds = roc_curve(test_labels, test_predict)
    
    plt.plot(fpr,tpr) 
    plt.title('ROC-Curve')
    plt.axis([0,1,0,1]) 
    plt.xlabel('false positive rate') 
    plt.ylabel('true positive rate') 
    plt.savefig(dir_name + '/ROC.png')    
    plt.show()
    plt.clf()
    
    print("Plot Confusion matrix...")
    test_predict_labels = np.where(test_predict>0.5, 1, 0)
        
    conf_matrix = confusion_matrix(test_labels, test_predict_labels)
    plt.title('Confusion Matrix')
    plt.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    # TODO: double check this
    plt.xlabel('predicted value') 
    plt.ylabel('true label') 
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
    plt.xlabel('predicted (r = false, g = true)') 
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
    '''
    Bias mitigation method, post-processing. Find ideal thresholds.
    
    :param model: trained model 
    :param val_predict: validation predictions
    :param val_labels: validation labels
    
    :return Tuple with optimal threshold of the four different methods:
            same amount of predicted true values, same amount of predicted false values, 
            same amount of predicted total values, same equal odds
            
            The second part of the tuple is the value of the specified optimization at each 
            threshold between 0 and 1
            
            The third part is the range of numbers between 0 and 1 that we consider 
            for the optimization
    '''    
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
    '''
    Bias mitigation method, post-processing. Test model with threshold change
    
    :param model: trained model 
    :param ds_val: validation predictions
    :param val_batches: validation labels
    :param test_predict: test predictions
    :param test_labels: test labels
    :param dir_name: directory to save results
    
    :return N/A
    '''  
    
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
        
        
def kfoldCrossValidation(path_female, path_male, model, metric_list, lr_schedule, image_size, class_weight, folds=5, epochs=20, learning_rate=5e-4, batch_size=128, shuffle_buffer=100):
    '''
    Performs k-fold crossvalidation 
    
    :param path_female: path to female training set 
    :param path_male: path to male training set
    :param model: untrained model
    :param metric_list: list of evaluation metrics
    :param lr_schedule: learning rate scheduler
    :param image_size: size of input image, should be selected according to model specification
    :param class_weight: weight per binary class
    :param folds: # of folds
    :param epochs: # of epochs
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param shuffle_buffer: ...
    
    
    :return history log containing results of cross validation
    '''      
    
    print("Load data...")
    female = glob.glob(path_female)
    male = glob.glob(path_male)

    data = []
    labels = []

    for i in female:   
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= image_size)
        image=np.array(image)
        data.append(image)
        labels.append(0)
    for i in male:   
        image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size= image_size)
        image=np.array(image)
        data.append(image)
        labels.append(1)
        
    data = np.array(data)
    labels = np.array(labels)

    list_data = list(zip(data, labels))

    random.shuffle(list_data)

    data, labels = zip(*list_data)

    data = np.array(data)
    labels = np.array(labels)
    
    print("Start crossvalidation...")
    
    kf = KFold(n_splits=folds, random_state=None, shuffle=True)
    
    weights = model.get_weights()
    
    history = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        print("Fold " + str(i) + "...")
        
        X_train, X_val = data[train_index], data[test_index]
        y_train, y_val = labels[train_index], labels[test_index]

        train_dataset = tf.data.Dataset.from_tensor_slices((np.array(X_train), np.array(y_train)))
        val_dataset = tf.data.Dataset.from_tensor_slices((np.array(X_val), np.array(y_val)))

        train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_dataset = train_dataset.with_options(options)
        val_dataset = val_dataset.with_options(options)

        model.set_weights(weights)

        model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule), 
                                   loss="binary_crossentropy", 
                                   metrics=metric_list)
        
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        earlystop_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
        
        results = model.fit(train_dataset, callbacks=[tensorboard_callback,earlystop_callback], class_weight=class_weight, epochs=epochs, validation_data=val_dataset)
        
        history.append(results.history)
        
    return history
        
    
def evaluateCrossValidation(dir_name, history):
    '''
    Evaluate crossvalidation from the history log file, prints and saves results to directory

    :param dir_name: directory where evaluation results should be saved
    :param history: history log for results of crossvalidation
    
    :return N/A
    ''' 
    
    from matplotlib import pyplot as plt
    accuracy = [element["accuracy"][-1] for element in history]
    loss = [element["loss"][-1] for element in history]

    tpr = [element["true_positive_rate"][-1] for element in history]
    tnr = [element["true_negative_rate"][-1] for element in history]
    fpr = [element["false_positive_rate"][-1] for element in history]
    fnr = [element["false_negative_rate"][-1] for element in history]

    ppv = [element["positive_predicted_value"][-1] for element in history]
    fdr = [element["false_discovery_rate"][-1] for element in history]
    npv = [element["negative_predicted_value"][-1] for element in history]
    fora = [element["false_omission_rate"][-1] for element in history]

    bppd = [element["binary_proportional_parity_diff"][-1] for element in history]
    beod = [element["binary_equalized_odds_diff"][-1] for element in history]
    bspd = [element["binary_specificity_parity_diff"][-1] for element in history]


    vaccuracy = [element["val_accuracy"][-1] for element in history]
    vloss = [element["val_loss"][-1] for element in history]


    vtpr = [element["val_true_positive_rate"][-1] for element in history]
    vtnr = [element["val_true_negative_rate"][-1] for element in history]
    vfpr = [element["val_false_positive_rate"][-1] for element in history]
    vfnr = [element["val_false_negative_rate"][-1] for element in history]

    vppv = [element["val_positive_predicted_value"][-1] for element in history]
    vfdr = [element["val_false_discovery_rate"][-1] for element in history]
    vnpv = [element["val_negative_predicted_value"][-1] for element in history]
    vfora = [element["val_false_omission_rate"][-1] for element in history]

    vbppd = [element["val_binary_proportional_parity_diff"][-1] for element in history]
    vbeod = [element["val_binary_equalized_odds_diff"][-1] for element in history]
    vbspd = [element["val_binary_specificity_parity_diff"][-1] for element in history]

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
       bppd=format(np.mean(bppd),".3f") + "+/-" + format(np.std(bppd),".3f"),
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
       vbppd=format(np.mean(vbppd),".3f") + "+/-" + format(np.std(vbppd),".3f"),
       vbeod=format(np.mean(vbeod),".3f") + "+/-" + format(np.std(vbeod),".3f"),
       vbspd=format(np.mean(vbspd),".3f") + "+/-" + format(np.std(vbspd),".3f")
    ))

    print("Plots...")

    plt.title('Model Accuracy')
    for element in history:
        plt.plot(element['accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/accuracy.png")
    plt.show()
    plt.clf()

    plt.title('Model Loss')
    for element in history:
        plt.plot(element['loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/loss.png")
    plt.show()
    plt.clf()

    plt.title('Model Binary Proportional Parity Diff')
    for element in history:
        plt.plot(element['binary_proportional_parity_diff'])
    plt.ylabel('binary proporitonal parity diff')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/proportional_parity.png")
    plt.show()

    plt.clf()

    plt.title('Model Binary Equalized Odds Diff')
    for element in history:
        plt.plot(element['binary_equalized_odds_diff'])
    plt.ylabel('binary equalized odds diff')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/equalized_odds.png")
    plt.show()
    plt.clf()

    plt.title('Model Validation Accuracy')
    for element in history:
        plt.plot(element['val_accuracy'])
    plt.ylabel('validation accuracy')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/val_accuracy.png")
    plt.show()
    plt.clf()

    plt.title('Model Validation Loss')
    for element in history:
        plt.plot(element['val_loss'])
    plt.ylabel('validation loss')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/val_loss.png")
    plt.show()
    plt.clf()

    plt.title('Model Validation Binary Proportional Parity Diff')
    for element in history:
        plt.plot(element['val_binary_proportional_parity_diff'])
    plt.ylabel('validation binary proportional parity diff')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/val_proporitional_parity.png")
    plt.show()
    plt.clf()

    plt.title('Model Validation Binary Equalized Odds Diff')
    for element in history:
        plt.plot(element['val_binary_equalized_odds_diff'])
    plt.ylabel('validation binary equalized odds diff')
    plt.xlabel('epoch')
    plt.savefig(dir_name + "/val_equalized_odds.png")
    plt.show()
    plt.clf()

    
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
    
    Binary proportional parity diff ((tp+fp)/(tp+fp+tn+fn))-((tn+fn)/(tp+fp+tn+fn)): $bppd
    
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
    
    Binary proportional parity diff ((tp+fp)/(tp+fp+tn+fn))-((tn+fn)/(tp+fp+tn+fn)): $vbppd
    
    Binary equalized odds diff (tp/(tp+fn))-(tn/(tn+fp)): $vbeod
    Binary specificity parity diff tn/(tn+fp)-tp/(tp+fn): $vbspd
    """)