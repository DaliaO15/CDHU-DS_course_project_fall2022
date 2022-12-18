import tensorflow as tf
import data.imageReading as ir
from fairness import fairnessMetrics as fm
from model import baseModel as bm
from model import utils as utils
from model import biasMitigation as mit
from model import transferLearning as tl
from model import evaluation as ev
import datetime
import os

preprocess_input = tf.keras.applications.vgg16.preprocess_input

image_size = (224,224)
batch_size = 2
epochs = 60
(ds_train, ds_val, ds_test, count_classes) = ir.readData("../museumFaces", image_size, batch_size, preprocess_input)

base_model = bm.build_model()

utils.train_model(base_model, epochs, ds_train, ds_val, class_weight)

print("Make folder...")
dir_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_Test"
os.mkdir(dir_name)

utils.saveModel(base_model, dir_name + "/final_model.h5")

test_predict, test_labels, dir_name = ev.testModel(model, ds_test, dir_name)
ev.testModelWithThresholdChange(model, ds_val, test_predict, test_labels, dir_name)


