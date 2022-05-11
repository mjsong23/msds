from tensorflow.keras import datasets, layers, models
import re
import os
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import sklearn
import copy
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# list out category names
categories = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Residential', 'River',
              'SeaLake']
images_fp = {}
filepath_df = []
label_ohe = {}
for i, label in enumerate(categories):
    ohe = np.zeros(8)
    ohe[i - 1] = 1
    label_tensor = tf.convert_to_tensor(ohe, dtype='float32')
    label_ohe[label] = label_tensor

# for folder in categories:
#     images_fp[folder] = tf.data.Dataset.list_files(f'miniSAT/{folder}/*', shuffle=False)

# create dictionary of filenames for each category
timer = time.time()
for folder in categories:
    # images_fp[folder] = glob.glob(f'miniSAT\\{folder}\\*')
    images_fp[folder] = glob.glob(f'EuroSAT\\2750\\{folder}\\*')
print("individual filepath read: ", time.time() - timer)

timer = time.time()
# create list of files
# filepath_df = glob.glob('miniSAT\\*\\*')
filepath_df = glob.glob('EuroSAT\\2750\\*\\*')
print("filepath read: ", time.time() - timer)

# find class sizes
timer = time.time()
cl_size = []
for folder in images_fp:
    cl_size.append(len(images_fp[folder]))
print("class size calculation", time.time() - timer)

# # train and test sizes
cl_size = np.asarray(cl_size).astype('int32')
train_size = 0.8 * cl_size
test_size = cl_size - train_size

# generate list of numeric class labels
class_label = np.repeat(np.arange(len(train_size)), cl_size)
# train_class_label = tf.repeat(np.arange(len(train_size)), repeats=train_size, axis=0)
# test_class_label = tf.repeat(np.arange(len(test_size)), repeats=test_size, axis=0)

# one hot encoding of class labels
# class_OHE = to_categorical(class_label)

# train/test split, rewrite with dummy variables for y_train/y_test since we will recalculate it later
x_train_list, x_test_list, y_train_label, y_test_label = sklearn.model_selection.train_test_split(
    filepath_df, class_label, train_size=0.8, stratify=class_label, random_state=0xDEADBEEF)
# y_train = to_categorical(y_train_label)
# y_test = to_categorical(y_test_label)

x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_list)
x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_list)

# cast lists into ndarrays for boolean indexing for class by class PCA
x_train_array = np.array(x_train_list)
x_test_array = np.array(x_test_list)


# Process the image files
def process_image(file_path):
    parts = tf.strings.split(file_path, os.path.sep).numpy()
    segment = parts[-2]
    label_output = label_ohe[segment.decode("utf-8")]
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = img / 255  # Rescale the image
    return img, label_output


# Reshape the tensors after import
def set_shapes(image, tensor_label):
    image.set_shape((64, 64, 3))
    tensor_label.set_shape((8,))
    return image, tensor_label


# Import the image files using the filepaths from the x_train tensor & mapping
timer = time.time()
x_train = x_train_dataset.map(lambda x: tf.py_function(func=process_image,
                                                       inp=[x], Tout=(tf.float32, tf.float32)))
x_train = x_train.map(set_shapes)

# Do the same for x_test tensor
x_test = x_test_dataset.map(lambda x: tf.py_function(func=process_image,
                                                     inp=[x], Tout=(tf.float32, tf.float32)))
x_test = x_test.map(set_shapes)
print("mapping dataset", time.time() - timer)
# Batch the tensors

batch_size = 64
timer = time.time()
x_train_batched = x_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
x_test_batched = x_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
print("batching dataset", time.time() - timer)

# Repeat mapping and batching for the separate classes
sep_class_x = []
for i in range(len(categories)):
    sep_class = x_train_array[y_train_label == i]
    sep_class_tensor = tf.data.Dataset.from_tensor_slices(sep_class)
    sep_dataset = sep_class_tensor.map(lambda x: tf.py_function(func=process_image,
                                                                inp=[x], Tout=(tf.float32, tf.float32)))
    sep_dataset = sep_dataset.map(set_shapes)
    sep_class_x.append(sep_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE).cache())

# Obtain true classes after batching
timer = time.time()
y_train_true = [np.argmax(y) for x, y in x_train_batched.unbatch()]
y_test_true = [np.argmax(y) for x, y in x_test_batched.unbatch()]
print("obtaining true class labels", time.time() - timer)
timer = time.time()

# Choose optimizers
optimizer = Adam(learning_rate=0.001)
loss_fxn = tf.losses.CategoricalCrossentropy()


# Build model architecture 1
def model_trainer(train_set, test_set, cnn1_channel, cnn1_window, cnn2_channel, cnn2_window, h, opt, loss,
                  epoch, model_number, summary=False, checkpoints=False, savebest=False):
    time_1 = time.time()
    model = models.Sequential()
    model.add(layers.Conv2D(cnn1_channel, (cnn1_window, cnn1_window), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(cnn2_channel, (cnn2_window, cnn2_window), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(h, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))
    if summary:
        model.summary()
    # Initialize checkpoint to store past epochs
    if checkpoints:
        filename = f'Checkpoints\\{model_number}\\Saved_weights_model-{model_number}' + '-{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(filename, monitor="val_accuracy", verbose=0, save_weights_only=True,
                                     save_best_only=savebest)
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        history = model.fit(train_set, validation_data=test_set, epochs=epoch, callbacks=[checkpoint])
    else:
        model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
        history = model.fit(train_set, validation_data=test_set, epochs=epoch)
    print("Model Computation time: ", time.time() - time_1)

    return model, history


model_history_list = []

print("Architecture 0")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 45, optimizer,
                                        loss_fxn, 100, model_number=0, summary=True,
                                        checkpoints=True))

print("Architecture 1")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 8, 5, 16, 5, 52, optimizer,
                                        loss_fxn, 100, model_number=1, summary=True,
                                        checkpoints=True))

print("Architecture 2")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 16, 3, 32, 3, 22, optimizer,
                                        loss_fxn, 100, model_number=2, summary=True,
                                        checkpoints=True))

print("Architecture 3")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 16, 5, 32, 5, 23, optimizer,
                                        loss_fxn, 100, model_number=3, summary=True,
                                        checkpoints=True))

# # Obtain output of the flattened layer on the best models
# arc1_flat_layer = model_list[0].layers[4].output
# arc1_flat_model = tf.keras.Model(model_list[0].input, arc1_flat_layer)
# arc1_flattened_output = arc1_flat_model.predict(x_train_batched, batch_size=8)
#
# arc3_flat_layer = model_list[2].layers[4].output
# arc3_flat_model = tf.keras.Model(model_list[2].input, arc3_flat_layer)
# arc3_flattened_output = arc3_flat_model.predict(x_train_batched, batch_size=8)
#
# # PCA analysis on the flattened outputs of the training set
# pca_99 = PCA(n_components=0.99, svd_solver='full')
# # arc1_pca = pca_99.fit_transform(arc1_flattened_output)
# # arc1_h = arc1_pca.shape[1]
# #
# # arc3_pca = pca_99.fit_transform(arc3_flattened_output)
# # arc3_h = arc3_pca.shape[1]

print("Architecture 4")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 2276, optimizer,
                                        loss_fxn, 100, model_number=4, summary=True,
                                        checkpoints=True))
print("Architecture 5")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 16, 3, 32, 3, 1941, optimizer,
                                        loss_fxn, 100, model_number=5, summary=True, checkpoints=True))

# Calculate large hidden layer h by doing PCA on each class separately
# arc1_sep_pca = []
# for i in range(len(categories)):
#     separate_output = arc1_flat_model.predict(sep_class_x[i], batch_size=8)
#     sep_pca = pca_99.fit_transform(separate_output)
#     arc1_sep_pca.append(sep_pca.shape[1])
# arc1_sep_pca_sum = sum(arc1_sep_pca)
#
# arc3_sep_pca = []
# for i in range(len(categories)):
#     separate_output = arc3_flat_model.predict(sep_class_x[i], batch_size=8)
#     sep_pca = pca_99.fit_transform(separate_output)
#     arc3_sep_pca.append(sep_pca.shape[1])
# arc3_sep_pca_sum = sum(arc3_sep_pca)

print("Architecture 6")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 9531, optimizer,
                                        loss_fxn, 100, model_number=6, summary=True, checkpoints=True, savebest=True))

print("Architecture 7")
model_history_list.append(model_trainer(x_train_batched, x_test_batched, 16, 3, 32, 3, 9072, optimizer,
                                        loss_fxn, 100, model_number=7, summary=True, checkpoints=True, savebest=True))

model_list, history_list = zip(*model_history_list)
model_list = list(model_list)
history_list = list(history_list)

def acc_plot(train_acc, test_acc, num_epochs, start, spacing, title="Accuracy by Epoch"):
    x_tick_range = [x for x in range(start, num_epochs, spacing)]
    # MSE by epoch, zoomed in
    test = np.array(test_acc)
    # 90% confidence interval
    margin = 1.6*np.sqrt((test*(1-test)/4500))
    ci_upper = test+margin
    ci_lower = test-margin
    plt.figure(figsize=(8, 6))
    plt.style.use("seaborn-darkgrid")
    plt.plot(range(start, len(train_acc)), train_acc[start:], color="cornflowerblue", linewidth=2, label="Train")
    plt.plot(range(start, len(test_acc)), test_acc[start:], color="darkorange", linestyle="--", linewidth=2,
             label="Test")
    plt.fill_between(range(start, len(test_acc)), ci_lower[start:], ci_upper[start:], color='powderblue', alpha=1)
    plt.title(title, fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xlabel("Epoch #", fontsize=14)
    plt.xticks(x_tick_range, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.show()


def conf_matrix_perc(true, predicted, title):
    plt.figure()
    cm = confusion_matrix(true, predicted, normalize='true')  # number correct/number in true class
    cm = np.around(cm * 200) / 200  # Round to nearest .5%
    cmap = copy.copy(plt.get_cmap("crest"))
    cmap.set_under('#808080')
    sns.heatmap(cm, annot=True, fmt='.1%', cmap=cmap, vmin=1e-5)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title(title)


# load checkpoints with model.load_weights(checkpoint_filepath)
num_ep = 100
spacing = 10

# Architecture 0
train_accuracy = history_list[0].history['accuracy']
test_accuracy = history_list[0].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 100, 0, spacing, title="Architecture 0 Accuracy")
acc_plot(train_accuracy, test_accuracy, 100, 40, 5, title="Architecture 0 Accuracy, After 40 Epochs")
# best epoch = 90
curr_model = 0
m = 90
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 0 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 0 Test Set Confusion Matrix")
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# Architecture 1
train_accuracy = history_list[1].history['accuracy']
test_accuracy = history_list[1].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 1 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 30, spacing, title="Architecture 1 Accuracy, After 30 Epochs")
# Best epoch = 80
# load weights and get predictions
curr_model = 1
m = 80
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 1 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 1 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))


#
#
# Architecture 2
train_accuracy = history_list[2].history['accuracy']
test_accuracy = history_list[2].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 2 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 30, spacing, title="Architecture 2 Accuracy, After 30 Epochs")
# Best epoch = 85
# load weights and get predictions
curr_model = 2
m = 85
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 2 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 2 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# Architecture 3
train_accuracy = history_list[3].history['accuracy']
test_accuracy = history_list[3].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 3 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 20, spacing, title="Architecture 3 Accuracy, After 20 Epochs")
# Best epoch = 77
# load weights and get predictions
curr_model = 3
m = 77
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 3 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 3 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

#
#
# Architecture 4
train_accuracy = history_list[4].history['accuracy']
test_accuracy = history_list[4].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 4 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 10, spacing, title="Architecture 4 Accuracy, After 10 Epochs")
# best epoch 68
curr_model = 4
m = 68
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 4 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 4 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))



# Architecture 5
train_accuracy = history_list[5].history['accuracy']
test_accuracy = history_list[5].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 5 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 20, spacing, title="Architecture 5 Accuracy, After 20 Epochs")
# best epoch = 50
curr_model = 5
m = 50
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 5 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 5 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# Architecture 6
train_accuracy = history_list[6].history['accuracy']
test_accuracy = history_list[6].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 6 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 25, spacing, title="Architecture 6 Accuracy, After 25 Epochs")
# best epoch = 61
curr_model = 6
m = 61
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 5 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 5 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))


# Architecture 7
train_accuracy = history_list[7].history['accuracy']
test_accuracy = history_list[7].history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Architecture 7 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 15, spacing, title="Architecture 7 Accuracy, After 15 Epochs")
# optimal epoch is about epoch 34
curr_model =7
m = 35
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
model_list[curr_model].load_weights(filepath)
y_train_pred = np.argmax(model_list[curr_model].predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(model_list[curr_model].predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 1 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 1 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))


# Best model = model 0
# Rerun with different train/test splits
x_train_list, x_test_list, y_train_label, y_test_label = sklearn.model_selection.train_test_split(
    filepath_df, class_label, train_size=0.8, stratify=class_label)
# y_train = to_categorical(y_train_label)
# y_test = to_categorical(y_test_label)

x_train_dataset = tf.data.Dataset.from_tensor_slices(x_train_list)
x_test_dataset = tf.data.Dataset.from_tensor_slices(x_test_list)

# cast lists into ndarrays for boolean indexing for class by class PCA
x_train_array = np.array(x_train_list)
x_test_array = np.array(x_test_list)

timer = time.time()
x_train = x_train_dataset.map(lambda x: tf.py_function(func=process_image,
                                                       inp=[x], Tout=(tf.float32, tf.float32)))
x_train = x_train.map(set_shapes)

# Do the same for x_test tensor
x_test = x_test_dataset.map(lambda x: tf.py_function(func=process_image,
                                                     inp=[x], Tout=(tf.float32, tf.float32)))
x_test = x_test.map(set_shapes)
print("mapping dataset", time.time() - timer)
# Batch the tensors

batch_size = 64
timer = time.time()
x_train_batched = x_train.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
x_test_batched = x_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
print("batching dataset", time.time() - timer)

timer = time.time()
y_train_true = [np.argmax(y) for x, y in x_train_batched.unbatch()]
y_test_true = [np.argmax(y) for x, y in x_test_batched.unbatch()]
print("obtaining true class labels", time.time() - timer)
timer = time.time()


# Model 8 - Same architecture as model 0
print("Model 8 - Architecture 0")
new_model, new_history = model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 45, optimizer,
                                        loss_fxn, 100, model_number=8, summary=True,
                                        checkpoints=True)

# Accuracy
train_accuracy = new_history.history['accuracy']
test_accuracy = new_history.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, num_ep, 0, spacing, title="Model 8 Accuracy")
acc_plot(train_accuracy, test_accuracy, num_ep, 15, spacing, title="Model 8 Accuracy, After 15 Epochs")
# optimal epoch is about epoch 89
curr_model =8
m = 80
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
new_model.load_weights(filepath)
y_train_pred = np.argmax(new_model.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(new_model.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Model 8 Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Model 8 Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

#improvements?
# lower learning rate
optimizer = Adam(learning_rate=0.0001)

print("Model 9 - Architecture 0")
improved_model, improved_history = model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 45, optimizer,
                                        loss_fxn, 300, model_number=9, summary=True,
                                        checkpoints=True)
train_accuracy = improved_history.history['accuracy']
test_accuracy = improved_history.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 300, 0, 20, title="Lower Learning Rate Model Accuracy")
acc_plot(train_accuracy, test_accuracy, 300, 150, 10, title="Lower Learning Rate Model Accuracy, After 150 Epochs")
# optimal epoch is about epoch 220
curr_model = 9
m = 220
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model.load_weights(filepath)
y_train_pred = np.argmax(improved_model.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Lower Learning Rate Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Lower Learning Rate Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# larger batch size to increase smoothness
x_train_batched = x_train_batched.unbatch().batch(128).prefetch(tf.data.experimental.AUTOTUNE).cache()
x_test_batched = x_test_batched.unbatch().batch(128).prefetch(tf.data.experimental.AUTOTUNE).cache()

optimizer = Adam(learning_rate=0.001)
improved_model_2, improved_history_2 = model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 45, optimizer,
                                        loss_fxn, 200, model_number=10, summary=True,
                                        checkpoints=True)

train_accuracy = improved_history_2.history['accuracy']
test_accuracy = improved_history_2.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 150, 0, 10, title="Larger Batch Size Model Accuracy")
acc_plot(train_accuracy, test_accuracy, 150, 20, 10, title="Larger Batch Size Model Accuracy, After 20 Epochs")
# optimal epoch is about epoch 90
curr_model = 10
m = 90
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model_2.load_weights(filepath)
y_train_pred = np.argmax(improved_model_2.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model_2.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Lower Learning Rate Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Lower Learning Rate Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# both larger batch size and slower learning rate
optimizer = Adam(learning_rate=0.0005)
improved_model_3, improved_history_3 = model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 45, optimizer,
                                        loss_fxn, 300, model_number=11, summary=True,
                                        checkpoints=True)
train_accuracy = improved_history_3.history['accuracy']
test_accuracy = improved_history_3.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 300, 0, 20, title="Combined Model Accuracy")
acc_plot(train_accuracy, test_accuracy, 300, 150, 10, title="Combined Model Accuracy, After 150 Epochs")
# optimal epoch is about epoch 200
curr_model = 11
m = 200
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model_3.load_weights(filepath)
y_train_pred = np.argmax(improved_model_3.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model_3.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Combined Model Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Combined Model Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# Architecture 4 with larger batch and slower learning rate
print("Improved Architecture 4")
improved_model_4, improved_history_4 = model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 2276, optimizer,
                                        loss_fxn, 300, model_number=12, summary=True,
                                        checkpoints=True)
train_accuracy = improved_history_4.history['accuracy']
test_accuracy = improved_history_4.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 300, 0, 20, title="Architecture 4 Improved Accuracy")
acc_plot(train_accuracy, test_accuracy, 300, 80, 10, title="Architecture 4 Improved, After 80 Epochs")
# optimal epoch is about epoch 90
curr_model = 12
m = 90
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model_4.load_weights(filepath)
y_train_pred = np.argmax(improved_model_4.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model_4.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Architecture 4 Improved Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Architecture 4 Improved Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# Change back to 64 batch
x_train_batched = x_train_batched.unbatch().batch(64).prefetch(tf.data.experimental.AUTOTUNE).cache()
x_test_batched = x_test_batched.unbatch().batch(64).prefetch(tf.data.experimental.AUTOTUNE).cache()


# Architecture 5 with Adagrad
adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01)
improved_model_5, improved_history_5 = model_trainer(x_train_batched, x_test_batched, 16, 3, 32, 3, 1941, adagrad,
                                        loss_fxn, 150, model_number=13, summary=True, checkpoints=True)

train_accuracy = improved_history_5.history['accuracy']
test_accuracy = improved_history_5.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 150, 0, 20, title="Architecture 5 Adagrad Accuracy")
acc_plot(train_accuracy, test_accuracy, 150, 60, 10, title="Architecture 5 Adagrad, After 60 Epochs")
# optimal epoch is about epoch 90
curr_model = 13
m = 130
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model_5.load_weights(filepath)
y_train_pred = np.argmax(improved_model_5.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model_5.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Architecture 5 Adagrad Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Architecture 5 Adagrad Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))

# rerun with slower learning rate with existing weights
file = 'Checkpoints\\14\\Saved_weights_model-14' + '-{epoch:02d}.hdf5'
cp = ModelCheckpoint(file, monitor="val_accuracy", verbose=0, save_weights_only=True,
                                     save_best_only=False)
adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.001)
improved_model_5.compile(optimizer=adagrad, loss=loss_fxn, metrics=['accuracy'])
improved_history_5_2 = improved_model_5.fit(x_train_batched, validation_data=x_test_batched, epochs=100, callbacks=[cp])

train_accuracy = improved_history_5_2.history['accuracy']
test_accuracy = improved_history_5_2.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 100, 0, 20, title="Architecture 5 Adagrad Rerun Accuracy")
acc_plot(train_accuracy, test_accuracy, 100, 60, 10, title="Architecture 5 Adagrad, After 60 Epochs")
# optimal epoch 67
curr_model = 14
m = 67
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model_5.load_weights(filepath)
y_train_pred = np.argmax(improved_model_5.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model_5.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Architecture 5 Adagrad Rerun Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Architecture 5 Adagrad Rerun Test Set Confusion Matrix")
# Trainacc/test at m*
print("Training Accuracy: ", sum(y_train_pred == y_train_true)/len(y_train_pred))
print("Test Accuracy: ", sum(y_test_pred == y_test_true)/len(y_test_pred))


# Model 0 with adagrad.
improved_model_6, improved_history_6 = model_trainer(x_train_batched, x_test_batched, 8, 3, 16, 3, 45, adagrad,
                                        loss_fxn, 300, model_number=15, summary=True,
                                        checkpoints=True)
train_accuracy = improved_history_6.history['accuracy']
test_accuracy = improved_history_6.history['val_accuracy']
acc_plot(train_accuracy, test_accuracy, 300, 0, 30, title="Architecture 0 Adagrad Rerun Accuracy")
acc_plot(train_accuracy, test_accuracy, 300, 60, 30, title="Architecture 0 Adagrad, After 60 Epochs")
curr_model = 15
m = 250
filepath = f'Checkpoints\\{curr_model}\\Saved_weights_model-{curr_model}-{m}.hdf5'
improved_model_5.load_weights(filepath)
y_train_pred = np.argmax(improved_model_5.predict(x_train_batched, batch_size=8), axis=1)
y_test_pred = np.argmax(improved_model_5.predict(x_test_batched, batch_size=8), axis=1)
conf_matrix_perc(y_train_true, y_train_pred, "Architecture 5 Adagrad Rerun Training Set Confusion Matrix")
conf_matrix_perc(y_test_true, y_test_pred, "Architecture 5 Adagrad Rerun Test Set Confusion Matrix")