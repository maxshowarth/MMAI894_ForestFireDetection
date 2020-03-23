import numpy as np
from keras.callbacks import EarlyStopping

import model_plot_utils as mpu
from blob_utils import *
from image_load_split_augment import load_augmented_dataset
from model_build import build_vgg_trainable, build_mobilenetv2, build_nasnet, build_resnetv2, build_efficientnet, \
    build_watts
from model_evaluation import *
from train import *

BUCKET_NAME = "citric-inkwell-268501"

retrainingConfigurations = [["allBlock5", ['block5_conv1', 'block5_conv2', 'block5_conv3']],
                            ["allBlock4", ['block4_conv1', 'block4_conv2', 'block4_conv3']],
                            ["block5_conv1", ['block5_conv1']],
                            ["allBlocks_conv1", ['block5_conv1', 'block4_conv1', 'block3_conv1', 'block2_conv1', 'block1_conv1']],
                            ["allBlocks_conv123", ['block5_conv1', 'block5_conv2', 'block5_conv3','block4_conv1', 'block4_conv2', 'block3_conv3', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block2_conv1', 'block2_conv2', 'block2_conv3', 'block1_conv1', 'block1_conv2', 'block1_conv3']]
                            ]

# for retrainingConfiguration in retrainingConfigurations:
#     nameOverride = "vgg16_experiments_{}".format(str(retrainingConfiguration[0]))
#     train_vgg16(nameOverride = nameOverride,layers_to_train = retrainingConfiguration[1])

savedModels = load_saved_models()
modelsToEvaluate = []
for model in savedModels.values():
    if "vgg16_experiments" in model[0]:
        modelsToEvaluate.append(model[0])

prediction(models_to_evaluate = modelsToEvaluate)