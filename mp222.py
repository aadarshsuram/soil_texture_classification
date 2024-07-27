import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button

# Define paths
train_dir = r'D:\mini\soil_photos\train'  # Update with your train directory path

# Check if the directory exists
if not os.path.exists(train_dir):
    print(f"Directory {train_dir} does not exist. Creating a sample dataset...")
    
    # Sample directory structure and images
    sample_dir = Path(r'D:\mini\soil_photos\train')  # Update with sample data directory path
    categories = ['clay', 'Sand', 'Silt']

    # Create directories
    for category in categories:
        os.makedirs(sample_dir / category, exist_ok=True)

    # Creating more sample images
    for category in categories:
        for i in range(100):  # Increase number of sample images
            img = Image.fromarray(np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8))
            img.save(sample_dir / category / f'{category}_{i}.jpg')
    
    train_dir = str(sample_dir)
    print(f"Sample dataset created at {train_dir}")
else:
    print(f"Directory {train_dir} found.")

# Image dimensions
img_width, img_height = 150, 150
batch_size = 8  # Reduce batch size

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split for validation
)

# Training data generator without repeat
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation data generator without repeat
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Build the model
model = Sequential([
    Input(shape=(img_width, img_height, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_soil_classifier.keras', save_best_only=True, monitor='val_loss', verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=None,  # Automatically set to the number of batches per epoch
    validation_data=validation_generator,
    validation_steps=None,  # Automatically set to the number of validation batches per epoch
    epochs=50,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Display final training and validation accuracy
final_train_accuracy = history.history['accuracy'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]
messagebox.showinfo("Training Completed", f"Final Training Accuracy: {final_train_accuracy:.2f}\nFinal Validation Accuracy: {final_val_accuracy:.2f}")

# Function to classify a new image, suggest crops, and discuss practical applications
def classify_soil(image_path):
    try:
        # Load the model
        model = load_model('best_soil_classifier.keras')
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(img_width, img_height))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)[0]

        # Get the class labels
        class_labels = {v: k for k, v in train_generator.class_indices.items()}
        predicted_class = class_labels[class_idx]

        # Display the image and the predicted class
        plt.imshow(img)
        plt.title(f'Predicted Soil Type: {predicted_class}')
        plt.axis('off')
        plt.show()

        # Crop suggestions based on soil type
        crop_suggestions = {
            'clay': "Recommended crops for clay soil: Wheat, Barley, Broccoli, Cabbage",
            'Sand': "Recommended crops for sand soil: Maize, Potatoes, Carrots",
            'Silt': "Recommended crops for silt soil: Soybeans, Peas, Alfalfa"
        }

        # Show classification result
        messagebox.showinfo("Soil Classification Result", f"The predicted soil type is: {predicted_class}\n\n{crop_suggestions.get(predicted_class, 'Soil type not recognized.')}")

        # Practical applications discussion
        print("\nPractical Applications:")
        print("- Soil texture classification can optimize irrigation and nutrient management strategies.")
        print("- Understanding soil types aids in crop rotation planning to maintain soil fertility.")
        print("- Soil classification supports environmental conservation efforts, such as erosion control.")
        print("- In construction and urban planning, knowledge of soil textures informs foundation design and infrastructure development.")

        return predicted_class

    except Exception as e:
        messagebox.showerror("Error", f"Error occurred: {str(e)}")

# Function to handle file selection and classification
def select_image_and_classify():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        classify_soil(file_path)

# Create GUI window
root = tk.Tk()
root.title("Soil Texture Classifier")

# Interface components
title_label = Label(root, text="Upload an Image of Soil", font=("Helvetica", 16))
title_label.pack(pady=10)

upload_button = Button(root, text="Upload Image", command=select_image_and_classify, font=("Helvetica", 14))
upload_button.pack(pady=20)

instructions_label = Label(root, text="Note: Supported image formats include JPG, JPEG, and PNG.", font=("Helvetica", 10))
instructions_label.pack()

# Run the GUI
root.mainloop()
