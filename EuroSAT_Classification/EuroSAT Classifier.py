# General Imports
import os
import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import seaborn as sns
import copy
from scipy.optimize import minimize

# DL Imports
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


# Preprocessing for transfer learning

(train_ds, val_ds, test_ds), metadata = tfds.load('eurosat/rgb', split=['train[:75%]', 'train[75%:95%]', 'train[95%:]'],
                                                  as_supervised=True, with_info=True)

val_classes = ['Industrial Buildings', 'Residential Buildings', 'Annual Crop', 'Permenant Crop', 'River',
               'SeaLake', 'HerbaceousVegetation', 'Highway', 'Pasture', 'Forest']
image_shape = metadata.features['image'].shape
num_classes = metadata.features['label'].num_classes

# Cache and shuffle the datasets
train_ds = train_ds.cache().shuffle(2000).prefetch(buffer_size=tf.data.AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Split training dataset into batches for each epoch
batch_size = 128
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

# Data Augmentation
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=image_shape),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Define model architecture
model = Sequential([

    data_augmentation,
    layers.Rescaling(1. / 255),
    layers.Conv2D(32, 3, padding='valid', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='valid', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='valid', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(.1),
    layers.Conv2D(256, 3, padding='valid', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Define Optimizer and compile
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# Fit the model
epochs = 200
rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                              patience=7, verbose=1)
estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                         verbose=1,
                                         restore_best_weights=True)

time_begin = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[rlronp, estop]
)
print("Model Computation time: ", time.time() - time_begin)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Test Accuracy')
plt.ylabel("Accuracy", fontsize=14)
plt.xlabel("Epoch #", fontsize=14)
plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.ylabel("Loss (Cross Entropy)", fontsize=14)
plt.xlabel("Epoch #", fontsize=14)
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()


def acc_plot(train_acc, test_acc, num_epochs, start, spacing, title="Accuracy by Epoch"):
    x_tick_range = [x for x in range(start, num_epochs, spacing)]
    # MSE by epoch, zoomed in
    test = np.array(test_acc)
    # 90% confidence interval
    margin = 1.6 * np.sqrt((test * (1 - test) / 4500))
    ci_upper = test + margin
    ci_lower = test - margin
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


acc_plot(train_acc=acc, test_acc=val_acc,
         num_epochs=len(acc), start=5, spacing=5, title="Accuracy by Epoch")

y_true = [y for x, y in test_ds]

test_set_size = test_ds.cardinality().numpy()
test_ds = test_ds.batch(test_set_size)
y_pred = np.argmax(model.predict(test_ds), axis=-1)

con_mat = tf.math.confusion_matrix(y_pred, y_true).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm,
                          index=val_classes,
                          columns=val_classes)

# Confusion matrix as percentages
figure = plt.figure(figsize=(8, 8))
cmap = copy.copy(plt.get_cmap("crest"))
cmap.set_under('#808080')
sns.heatmap(con_mat_df, annot=True, cmap=cmap)
plt.tight_layout()
plt.title('Test Set Confusion Matrix', fontsize=12)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()

print('Overall Test Accuracy: ', "{0:.0%}".format(np.sum(y_pred == y_true) / len(y_pred)))
# 95.5% training accuracy, 93.5% validation accuracy
# 94-95% test accuracy on held out cases