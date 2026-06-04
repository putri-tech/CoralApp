import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

# ==========================================
# 1. PENGATURAN DIREKTORI DATASET
# ==========================================
BASE_DIR = r"C:\Users\ASUS\Downloads\KULIAH_PENGOLAHAN_CITRA\dataset_coral_terpisah"

IMG_SIZE = (300, 300)
BATCH_SIZE = 4

# ==========================================
# 2. CALLBACKS
# ==========================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ==========================================
# 3. DATA GENERATOR
# ==========================================
print("=== Mempersiapkan Data Generator ===")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,

    zoom_range=0.2,

    brightness_range=[0.85, 1.15],

    horizontal_flip=True,
    vertical_flip=True,

    fill_mode='nearest',

    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# ==========================================
# 4. MEMBANGUN MODEL
# ==========================================
print("\n=== Membangun EfficientNetB0 ===")

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(300, 300, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 5. TRAINING TAHAP 1
# ==========================================
print("\n=== Training Tahap 1 ===")

EPOCHS_TAHAP1 = 10

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_TAHAP1,
    validation_data=val_generator,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

# ==========================================
# 6. FINE TUNING
# ==========================================
print("\n=== Training Tahap 2 (Fine Tuning) ===")

base_model.trainable = True

for layer in base_model.layers[:-120]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

EPOCHS_TAHAP2 = 30

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_TAHAP2,
    validation_data=val_generator,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

# ==========================================
# 7. LOAD MODEL TERBAIK
# ==========================================
print("\n=== Memuat Model Terbaik ===")

model.load_weights('best_model.h5')

# ==========================================
# 8. SIMPAN MODEL H5
# ==========================================
print("\n=== Menyimpan Model H5 ===")

model.save('model_coral_efficientnet.h5')

# ==========================================
# 9. KONVERSI KE TFLITE
# ==========================================
print("\n=== Mengonversi ke TFLite ===")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model_coral_efficientnet.tflite', 'wb') as f:
    f.write(tflite_model)

print("=== SELESAI ===")