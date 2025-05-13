import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
IMG_SIZE = 384
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 5

# --- DATASET PATHS ---
IMG_DIR = '/content/drive/MyDrive/IDRID_Diagonsis/train'
LABELS_PATH = '/content/drive/MyDrive/IDRID_Diagonsis/training_labels.xlsx'

# --- READ LABELS ---
df = pd.read_excel(LABELS_PATH)
df['Image name'] = df['Image name'].astype(str) + '.jpg'
df['Retinopathy grade'] = df['Retinopathy grade'].astype(int)

# --- SPLIT DATA ---
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Retinopathy grade'], random_state=42)

# --- CLAHE PREPROCESSING ---
def apply_clahe_rgb(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced_img

# --- AUGMENTATION PIPELINE ---
train_aug = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.Rotate(limit=15),
    A.Normalize(),
    ToTensorV2()
])

val_aug = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# --- DATA GENERATOR ---
class DRDataset(tf.keras.utils.Sequence):
    def __init__(self, dataframe, img_dir, augment=None, batch_size=BATCH_SIZE):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        images, labels = [], []
        for _, row in batch_df.iterrows():
            img = apply_clahe_rgb(os.path.join(self.img_dir, row['Image name']))
            label = row['Retinopathy grade']
            if self.augment:
                img = self.augment(image=img)['image']
                img = tf.cast(img, tf.float32)
                img = tf.transpose(img, perm=[1, 2, 0])  # CxHxW -> HxWxC
            images.append(img)
            labels.append(label)
        return tf.stack(images), tf.convert_to_tensor(labels)

# --- DATALOADERS ---
train_loader = DRDataset(train_df, IMG_DIR, augment=train_aug)
val_loader = DRDataset(val_df, IMG_DIR, augment=val_aug)

# --- FOCAL LOSS ---
def focal_loss(gamma=2., alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# --- MODEL: ConvNeXt-B ---
def build_model():
    base_model = ConvNeXtBase(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights='imagenet'
    )
    base_model.trainable = True

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

model = build_model()
model.compile(optimizer=Adam(2e-4), loss=focal_loss(gamma=2.0, alpha=0.5), metrics=['accuracy'])
model.summary()

# --- CALLBACKS ---
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ModelCheckpoint('best_dr_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

# --- TRAIN ---
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# --- PLOT ---
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("ConvNeXt-B DR Classification Performance (Focal Loss)")
plt.legend()
plt.grid()
plt.show()
