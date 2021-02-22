import numpy as np
import os, sys, time, re, gc
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.layers import Conv2D
from keras import backend as K
from pathlib import Path
from glob import glob

import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import ResNet50, ResNet101, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from utils.pruning_method import pruning_method_conv
from model.model_architectures import model_type

"""    GPU enable and enables running the script without errors    """
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    try: # either line 17 * 18 are used, or line 21 is solely used (line 21 doesn't need "for gpu in gpus:")
        # # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # # Allocating fixed memory.
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

########################################################################################################################
#                                            Functions BEGIN                                                           #
########################################################################################################################

""" Pruning """
def pruning_filters_conv(pruning_index, layer_to_prune, model_for_pruning, original_num_filters, method):
    # pruning_index = 0.05  # 0.05 = 5% of the filters are to be pruned
    pruning_amount = [int(original_num_filters[i] * pruning_index[i]) for i in range(len(original_num_filters))]

    model_pruned = pruning_method_conv(model_for_pruning, layer_to_prune, pruning_amount, method)

    sgd = SGD(lr=1e-3, decay=5e-4, momentum=0.9, nesterov=True)
    model_pruned.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

    return model_pruned


def top_k_accuracy(y_true, y_pred, k=1, tf_enabled=True):
    if tf_enabled:
        argsorted_y = tf.argsort(y_pred)[:,-k:]
        matches = tf.cast(tf.math.reduce_any(tf.transpose(argsorted_y) == tf.argmax(y_true, axis=1, output_type=tf.int32), axis=0), tf.float32)
        return tf.math.reduce_mean(matches).numpy()
    else:
        argsorted_y = np.argsort(y_pred)[:,-k:]
        return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


def prediction(model_pruned, x_val_paths, y_val_one_hot):
    y_pred = None
    for i, x_val_path in enumerate(x_val_paths):
        x_val = np.load(x_val_path).astype('float32')  # loaded as RGB
        x_val = resnet.preprocess_input(x_val)  # converted to BGR

        y_pred_sharded = model_pruned.predict(x_val, verbose=0, use_multiprocessing=True, batch_size=64, callbacks=None)

        try:
            y_pred = np.concatenate([y_pred, y_pred_sharded])
        except ValueError:
            y_pred = y_pred_sharded

        del x_val
        gc.collect()

        completed_percentage = (i + 1) * 100 / len(x_val_paths)
        if completed_percentage % 5 == 0:
            print("{:5.1f}% completed.".format(completed_percentage))

    return top_k_accuracy(y_val_one_hot, y_pred, k=1), top_k_accuracy(y_val_one_hot, y_pred, k=5)

########################################################################################################################
#                                              Functions END                                                           #
########################################################################################################################

##### Initialize environment #####
count = -1
layer_to_prune_original_model_conv = []
layer_to_prune_for_continuous_pruning_conv = []
original_num_filters = []
img_w, img_h = 224, 224
classes = 7

method = 'geometric_median_conv'
arch = 'resnet50'
dataset_ = 'imagenet'
dataset_path = 'D:/ImageNet/ILSVRC2012_img_val'
pruning_index_per = 0.10
epochs = 0
predict_batch_size = 10
VALIDATION_IMAGES = sum([len(files) for r, d, files in os.walk(dataset_path)])

path_imagenet_val_dataset = Path("D:/ImageNet/data/") # path/to/data/
dir_images = Path("D:/ImageNet/ILSVRC2012_img_val/validation/") # path/to/images/directory



##### Load data #####
x_val_paths = glob(str(path_imagenet_val_dataset / "x_val*.npy"))
# Sort filenames in ascending order
x_val_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
y_val = np.load(str(path_imagenet_val_dataset / "y_val.npy"))
y_val_one_hot = to_categorical(y_val, 1000)



##### Setting Model #####
model = ResNet50(weights='imagenet')
model.summary()



##### Check Conv2D Layers #####
for layer in model.layers:
    count = count + 1
    if isinstance(layer, Conv2D):
        layer_to_prune_original_model_conv.append(count)
        original_num_filters.append(layer.weights[0].shape[3])

pruning_index_temp = np.ones((len(layer_to_prune_original_model_conv),)) * pruning_index_per # 0.05 = 5% of the filters are to be pruned



##### For continuous pruning job #####
for layer_to_prune in range(0, len(layer_to_prune_original_model_conv)):
    print("Layer to prune: ", layer_to_prune_original_model_conv[layer_to_prune])
    if os.path.isdir('D:/ResNet50_Pruning/test_continuous_pruning_layer{}'.format(layer_to_prune_original_model_conv[layer_to_prune])) == False:
       os.mkdir('D:/ResNet50_Pruning/test_continuous_pruning_layer{}'.format(layer_to_prune_original_model_conv[layer_to_prune]))

    ##### Pruning each Conv2D Layer #####
    pruning_index = [pruning_index_temp[layer] if layer == layer_to_prune else 0 for layer in range(len(pruning_index_temp))]
    for i in range(1, int(1/pruning_index_per)):
        if i == 1:
            # prune & save conv layer
            model_pruned = pruning_filters_conv(pruning_index, layer_to_prune_original_model_conv, model, original_num_filters, method)
            # model_pruned.save('D:/ResNet50_Pruning/test_continuous_pruning_layer{}/{}_{}_after_prune_{}_{}%_{}epochs.h5'
            #                   .format(layer_to_prune_original_model_conv[layer_to_prune], arch, dataset_, method, pruning_index_per*100, epochs))

        else:
            # load model to prune
            model_pruned = load_model('D:/ResNet50_Pruning/test_continuous_pruning_layer{}/{}_{}_after_prune_{}_{}%_{}epochs.h5'
                                      .format(layer_to_prune_original_model_conv[layer_to_prune], arch, dataset_,method, pruning_index_per * 100 * (i - 1), epochs))

            # prune & save conv layer
            model_pruned = pruning_filters_conv(pruning_index, layer_to_prune_original_model_conv, model_pruned, original_num_filters, method)
            # model_pruned.save('D:/ResNet50_Pruning/test_continuous_pruning_layer{}/{}_{}_after_prune_{}_{}%_{}epochs.h5'
            #                       .format(layer_to_prune_original_model_conv[layer_to_prune], arch, dataset_, method, pruning_index_per*100*i, epochs))

        # del model_pruned
        # K.clear_session()

        #### Evaluation after pruning #####
        # model_pruned = load_model('D:/ResNet50_Pruning/test_continuous_pruning_layer{}/{}_{}_after_prune_{}_{}%_{}epochs.h5'
        #                           .format(layer_to_prune_original_model_conv[layer_to_prune], arch, dataset_, method, pruning_index_per*100*i, epochs))

        print("Start prediction:")
        top_1, top_5 = prediction(model_pruned, x_val_paths, y_val_one_hot)
        results = [top_1, top_5]

        np.savetxt('D:/ResNet50_Pruning/test_continuous_pruning_layer{}/{}_{}_after_prune_{}_{}%_{}epochs.csv'
                   .format(layer_to_prune_original_model_conv[layer_to_prune], arch, dataset_,method, pruning_index_per*100*i, epochs), results, delimiter=',')
        print('Test top-1 accuracy for pruned model: {}%'.format(results[0]))
        print('Test top-5 accuracy for pruned model: {}%'.format(results[1]))


        ##### Remove model from GPU memory #####
        del model_pruned
        K.clear_session()



















