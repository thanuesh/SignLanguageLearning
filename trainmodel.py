import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

img_size = 224  # Match with your model input (can also be 150 or 300)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% train, 20% validation
)

train_data = train_datagen.flow_from_directory(
    'data',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    'data',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save class indices as labels
label_map = train_data.class_indices
labels = [None] * len(label_map)
for k, v in label_map.items():
    labels[v] = k

# Save labels.txt
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
            MaxPooling2D(2, 2),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(128, activation='relu'),
            Dense(len(labels), activation='softmax')  # one output per label
        ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, validation_data=val_data, epochs=10)
model.save("keras_model.h5")

