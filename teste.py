import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from efficientnet.tfkeras import EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5
from ranger import Ranger
import torchvision.transforms as transforms

tf.config.run_functions_eagerly(True)

# Carregar o dataset
df = pd.read_csv("IA-deeplearning/plant-pathology/train.csv", index_col=0)
print(df.shape)
df.head()

train_set, valid_set = train_test_split(df, test_size=0.2, random_state=42)

print(train_set.shape)
print(valid_set.shape)
SOURCE = 'IA-deeplearning/plant-pathology/images/train'
VALID_DIR = 'IA-deeplearning/plant-pathology/temp/valid/'
TRAIN_DIR = 'IA-deeplearning/plant-pathology/temp/train/'

# Data Augmentation
training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(TRAIN_DIR, target_size=(600, 600), class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(VALID_DIR, target_size=(600, 600), class_mode='categorical')

# Define a política de augmentation usando ImageNetPolicy + Horizontal Flip
augmentation_policy = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
])

# Modelo Ensemble com EfficientNet-B2, B3, B4, B5
input_shape = (600, 600, 3)
base_models = [
    EfficientNetB2(input_shape=input_shape, include_top=False, weights='imagenet'),
    EfficientNetB3(input_shape=input_shape, include_top=False, weights='imagenet'),
    EfficientNetB4(input_shape=input_shape, include_top=False, weights='imagenet'),
    EfficientNetB5(input_shape=input_shape, include_top=False, weights='imagenet')
]

model_outputs = []
for base_model in base_models:
    base_model.trainable = False
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model_outputs.append(x)

# Combina as saídas dos modelos com Concatenate
combined_output = tf.keras.layers.Concatenate()(model_outputs)
x = tf.keras.layers.Dropout(0.4)(combined_output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
output = tf.keras.layers.Dense(4, activation='sigmoid')(x)  # BCEWithLogitsLoss requer sigmoid

model = tf.keras.models.Model(inputs=[model.input for model in base_models], outputs=output)

# Compilar o modelo com Ranger optimizer e BCEWithLogitsLoss
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=Ranger(),
    metrics=['accuracy']
)

# Callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=7)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("cnngpu.keras", save_best_only=True)
checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_loss.keras', save_best_only=True, monitor='val_loss', mode='min')
checkpoint_accuracy = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_accuracy.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Calcular steps_per_epoch e validation_steps
batch_size = 31
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Verificar se os generators têm pelo menos um lote de dados
if train_generator is not None and validation_generator is not None:
    if len(train_generator) > 0 and len(validation_generator) > 0:
        # forçar a execução na GPU
        with tf.device('/device:GPU:0'):
            history = model.fit(
                [train_generator] * len(base_models),
                epochs=20,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=[early_stopping_cb, checkpoint_cb, checkpoint_accuracy, checkpoint_loss],
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
plt.show()