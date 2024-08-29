import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing.image import load_img
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
tf.config.run_functions_eagerly(True)
df = pd.read_csv("IA-deeplearning/plant-pathology/train.csv", index_col=0)
train_set, valid_set = train_test_split(df, test_size=0.2, random_state=42)

SOURCE = 'IA-deeplearning/plant-pathology/images/train'
VALID_DIR = 'IA-deeplearning/plant-pathology/temp/valid/'
TRAIN_DIR = 'IA-deeplearning/plant-pathology/temp/train/'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



training_datagen = ImageDataGenerator(rescale = 1./255,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(TRAIN_DIR, target_size=(225, 150), class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(VALID_DIR, target_size=(225, 150), class_mode='categorical')

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(225, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])



# Compilar o modelo

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'], run_eagerly=True)

# Callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=7)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("cnngpu.keras", save_best_only=True)
checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_loss.keras', save_best_only=True, monitor='val_loss', mode='min')
checkpoint_accuracy = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_accuracy.keras', save_best_only=True, monitor='val_accuracy', mode='max')



# Calcular steps_per_epoch e validation_steps
batch_size = 31
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("nome da gpu", tf.test.gpu_device_name())
print(tf.test.is_gpu_available())
print("steps_per_epoch", steps_per_epoch)
print("validation_steps", validation_steps)

# print(TRAIN_DIR, VALID_DIR)
# Verificar se os generators têm pelo menos um lote de dados
if train_generator is not None and validation_generator is not None:
    if len(train_generator) > 0 and len(validation_generator) > 0:
        # forçar a execução na gpu
        with tf.device('/device:GPU:0'):
            history = model.fit(
                train_generator,
                epochs=50,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=[early_stopping_cb,checkpoint_cb,checkpoint_accuracy,checkpoint_loss],
                verbose=1
            )
    else:
        print("Error: The generators do not have any data.")
else:
    print("Error: One of the generators is None.")
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, val_acc, 'r', label='val_accuracy')
plt.plot(epochs, acc, 'b', label='val_loss')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.savefig('accuracy-avaliada.png')
