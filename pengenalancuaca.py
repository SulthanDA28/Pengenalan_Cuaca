import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

IMG_SIZE = (128, 128)  # Image size
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001


def build_model():
    train_path = "Dataset/training"
    test_path = "Dataset/test"
    model_file = "klasifikasi_cuaca.h5"

    # Data Preparation
    # Perubahan Ukuran Gambar dan Normalisasi
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    if os.path.exists(model_file):
        print(f"Membaca file model yang sudah dibuat: {model_file}")
        model = load_model(model_file)
        class_labels = list(test_generator.class_indices.keys())
        return model,class_labels
    else:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(train_generator.num_classes, activation='softmax')
        ])

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        # Latih Model
        model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=EPOCHS,
            verbose=1
        )

        # Simpan Model
        model.save(model_file)
        print(f"Model saved as {model_file}")

        # Evaluasi Model
        test_loss, test_acc = model.evaluate(test_generator, verbose=1)
        print(f"Test Accuracy: {test_acc:.2f}")

        # Classification Report
        test_generator.reset()
        y_true = test_generator.classes
        y_pred = np.argmax(model.predict(test_generator), axis=-1)
        class_labels = list(test_generator.class_indices.keys())
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_labels))
        return model,class_labels

# Predict a single image
def klasifikasi_cuaca(image_path,model,class_labels):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label
