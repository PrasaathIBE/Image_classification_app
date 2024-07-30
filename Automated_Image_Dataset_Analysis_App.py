import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Automated Image Classification App")

def load_data(data_dir, target_size=(128, 128)):
    data = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')  # Ensure 3 channels
                    img = img.resize(target_size)
                    img = np.array(img)
                    data.append(img)
                    labels.append(class_name)
                except Exception as e:
                    st.error(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels), class_names

# Use your specified directory path
data_dir = 'D:/01_For_Thanjai_Sir/Image_Classification/image_classification_dataset/Agricultural-crops'
data, labels, class_names = load_data(data_dir)

if len(data) == 0 or len(labels) == 0:
    st.error("No data found. Please check the dataset.")
else:
    st.write("Data Loaded Successfully!")
    st.write(f"Number of images: {len(data)}")
    st.write(f"Number of labels: {len(labels)}")
    st.write(f"Classes: {class_names}")

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_encoded = to_categorical(labels_encoded)

    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow(X_train, y_train, batch_size=32)
    validation_generator = datagen.flow(X_test, y_test, batch_size=32)

    def create_model(input_shape, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    input_shape = X_train.shape[1:]
    num_classes = len(class_names)
    model = create_model(input_shape, num_classes)
    model.summary()

    st.write("Training the model...")
    history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

    st.write("Model trained successfully!")

    st.write("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    cm = confusion_matrix(y_true_classes, y_pred_classes)

    st.write(f"Accuracy: {accuracy}")
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    st.pyplot(fig)

    st.write("Training and Validation Accuracy:")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

    st.write("Training and Validation Loss:")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)
