from tensorboard.backend.event_processing import event_accumulator as ea
import tensorflow as tf
from matplotlib import pyplot as plt

"""
Contributions: Christoph Nötzli
Comments: Christoph Nötzli 17/12-22
"""

def loadEventAccumulators(data_names, log_files_train, log_files_val, network="efficientnet"):
    """
    Load data from log files
    
    :param data_names: Name of result directory name
    :param log_files_train: List of training log file paths
    :param log_files_val: List of validation log file paths
    :param network: Either "efficientnet" or "xception"
    
    :return reloaded data from log files
    """
    efficientnet_dict = efficientnetDict()
    xception_dict = xceptionDict()
    validation_sets = []
    training_sets = []
    for data_name in data_names:
        if(network == "xception"):
            for element in xception_dict[data_name]:
                for file in log_files_train:
                    if(element in file):
                        training_sets.append(file)
                for file in log_files_val:
                    if(element in file):                
                        validation_sets.append(file)
        else:
            for element in efficientnet_dict[data_name]:
                for file in log_files_train:
                    if(element in file):
                        training_sets.append(file)
                for file in log_files_val:
                    if(element in file):                
                        validation_sets.append(file)

    event_acc_train = []
    for element in training_sets:
        train = ea.EventAccumulator(element)
        train.Reload()
        event_acc_train.append(train)

    event_acc_val = []
    for element in validation_sets:
        val = ea.EventAccumulator(element)
        val.Reload()
        event_acc_val.append(val)

    print("Number of training files: ", len(event_acc_train))
    print("Number of validation files: ", len(event_acc_train))
    
    return (event_acc_train, event_acc_val)

def createDataList(event_acc_train, event_acc_val):
    """
    Create a dictionary that has the metric of all logs in the same list
    
    :param event_acc_train: Training logs
    :param event_acc_val: Validation logs
    
    :return data_list with metrics of the logs in the same list
    """
    data_list = {}

    for train in event_acc_train:
        for tag in train.Tags()["tensors"]:
            if 'epoch' in tag:
                result = []
                for index, element in enumerate(train.Tensors(tag)):
                    result.append(tf.make_ndarray(element.tensor_proto))

                if "epoch_true_positives" in tag:
                    tag = "epoch_true_positives"
                if "epoch_false_positives" in tag:
                    tag = "epoch_false_positives"
                if "epoch_true_negatives" in tag:
                    tag = "epoch_true_negatives"
                if "epoch_false_negatives" in tag:
                    tag = "epoch_false_negatives"
                    
                # Change name to equal opportunity because of wrong tag name
                if "epoch_binary_equalized_odds_diff" in tag:
                    tag = "epoch_binary_equal_opportunity_diff"

                tag = tag.replace("epoch_", "")

                if not(tag in data_list):
                    data_list[tag] = []

                data_list[tag].append((range(1,len(result)+1), result))


    for val in event_acc_val:
        for tag in val.Tags()["tensors"]:
            if 'evaluation' in tag:
                result = []
                for index, element in enumerate(val.Tensors(tag)):
                    result.append(tf.make_ndarray(element.tensor_proto))

                if "evaluation_true_positives" in tag:
                    tag = "evaluation_true_positives_vs_iterations"
                if "evaluation_false_positives" in tag:
                    tag = "evaluation_false_positives_vs_iterations"
                if "evaluation_true_negatives" in tag:
                    tag = "evaluation_true_negatives_vs_iterations"
                if "evaluation_false_negatives" in tag:
                    tag = "evaluation_false_negatives_vs_iterations"
                    
                # Change name to equal opportunity because of wrong tag name
                if "evaluation_binary_equalized_odds_diff" in tag:
                    tag = "evaluation_binary_equal_opportunity_diff_vs_iterations"

                tag = tag.replace("evaluation_", "")
                tag = tag.replace("_vs_iterations", "")

                if not(tag in data_list):
                    data_list[tag] = []

                data_list[tag].append((range(1,len(result)+1), result))
                
    return data_list

    
def efficientnetDict():
    """
    Dict that links the result directories of the EfficientNet experiments to the corresponding log files
    
    :return Dict with linked results and logs
    """
    return {
        "20221212_185715_EfficientNet_Museum": ["20221212-184604"],
        "20221212_195519_kFoldCrossValidation_Museum_effnet_baseline": ["20221212-190206","20221212-191246","20221212-192358","20221212-193425","20221212-194541"],
        "20230103_175356_EfficientNet_Museum_re": ["20230103-174204"],
        "20230103_185007_kFoldCrossValidation_Museum_re_effnet": ["20230103-175422","20230103-180536","20230103-181603","20230103-182753","20230103-183822"],
        "20221213_181748_EfficientNet_Museum_aug": ["20221213-180415"],
        "20221213_191824_kFoldCrossValidation_Museum_aug_effnet": ["20221213-181816","20221213-183007","20221213-184128","20221213-185506","20221213-190414"],
        "20221213_204427_EfficientNet_FairFace": ["20221213-191829"],
        "20221213_205623_EfficientNet_transfer": ["20221213-205237"],
        "20221213_221541_EfficientNet_FairFace_re": ["20221213-213137"],
        "20221213_222455_EfficientNet_transfer_re": ["20221213-222109"],
        "20221214_133249_EfficientNet_FairFace_aug": ["20221214-105744"],
        "20221214_135251_EfficientNet_transfer_aug": ["20221214-134837"],
        "20221214_140453_EfficientNet_transfer_re_aug": ["20221214-140102"],
        "20221216_134535_kFoldCrossValidation_EfficientNet": ["20221216-133205","20221216-133447","20221216-133724","20221216-134014","20221216-134315"],
        "20221216_140029_kFoldCrossValidation_EfficientNet_aug": ["20221216-134551","20221216-134833","20221216-135149","20221216-135427","20221216-135758"],
        "20221216_140651_kFoldCrossValidation_EfficientNet_re": ["20221216-134945","20221216-135251","20221216-135649","20221216-140028","20221216-140400"],
        "20221216_142344_kFoldCrossValidation_EfficientNet_re_aug": ["20221216-140923","20221216-141213","20221216-141502","20221216-141803","20221216-142057"]
    }
    
def xceptionDict():
    """
    Dict that links the result directories of the Xception experiments to the corresponding log files
    
    :return Dict with linked results and logs
    """
    return {
        "20221213_104527_Xception_Museum": ["20221213-103916"],
        "20221213_110945_kFoldCrossValidation_Museum_xcep_baseline": ["20221213-104822","20221213-105236","20221213-105654","20221213-110108","20221213-110521"],
        "20221213_112138_Xception_Museum_re": ["20221213-111718"],
        "20221213_114659_kFoldCrossValidation_Museum_re_xcep": ["20221213-112554","20221213-113006","20221213-113416","20221213-113828","20221213-114245"],
        "20221213_115308_Xception_Museum_aug": ["20221213-114803"],
        "20221213_122549_kFoldCrossValidation_Museum_aug_xcep": ["20221213-115649","20221213-120105","20221213-120523","20221213-121448","20221213-122144"],
        "20221213_134120_Xception_FairFace": ["20221213-123022"],
        "20221213_162449_Xception_FairFace_re": ["20221213-150142"],
        "20221214_084253_Xception_transfer_re": ["20221214-084035"],
        "20221214_092855_kFoldCrossValidation_Xception_re": ["20221214-085259","20221214-085814","20221214-090652","20221214-091214","20221214-091659"],
        "20221214_112623_Xception_transfer": ["20221214-112338"],
        "20221214_120745_kFoldCrossValidation_Xception": ["20221214-113325","20221214-113858","20221214-114349","20221214-114845","20221214-115408"],
        "20221214_154717_Xception_FairFace_aug": ["20221214-120929"],
        "20221214_162902_Xception_transfer_aug": ["20221214-162613"],
        "20221214_171123_kFoldCrossValidation_Xception_aug": ["20221214-163047","20221214-163712","20221214-164603","20221214-165111","20221214-170010"],
        "20221215_105133_Xception_transfer_re_aug": ["20221215-104727"],
        "20221215_113456_kFoldCrossValidation_Xception_re_aug": ["20221215-105756","20221215-110652","20221215-111302","20221215-111919","20221215-112733"]
    }