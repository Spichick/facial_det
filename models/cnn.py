# use colab to train
'''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("msambare/fer2013")

print("Path to dataset files:", path)
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense

train_dir = "/kaggle/input/fer2013/train"
test_dir  = "/kaggle/input/fer2013/test"

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',
    image_size=(48, 48),
    batch_size=64
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    image_size=(48, 48),
    batch_size=64
)

# Normalize
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))
val_ds   = val_ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

# Define mini-Xception
def mini_xception(input_shape=(48,48,1), num_classes=7):
    model = Sequential([
        Conv2D(8, (3,3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(8, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        SeparableConv2D(16, (3,3), activation='relu', padding='same'),
        SeparableConv2D(16, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        SeparableConv2D(32, (3,3), activation='relu', padding='same'),
        SeparableConv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = mini_xception()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_ds, validation_data=val_ds, epochs=25)

# Save
model.save("/kaggle/working/mini_xception.h5")
print("✅ Model saved to /kaggle/working/mini_xception.h5")
