import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Step 1: Load Images and Assign Labels
def load_dataset(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Adjust the size as needed
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Step 2: Load and Split the Dataset
dataset_folder = 'E:\AI\Deep Learning\Projects\Object detetions\Drowsiness\Main Project\dataset2\Train'  # Change this to your dataset path
X, Y = load_dataset(dataset_folder)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train = (y_train == 'Drowsy').astype(int)
y_test = (y_test == 'Drowsy').astype(int)


# Step 3: Normalize Pixel Values
x_train = x_train / 255.0
x_test = x_test / 255.0



# Step 4: Verify the Shapes
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


def display_images(images, labels, num_images=20):
    fig, axes = plt.subplots(2, 10, figsize=(15, 6))
    for i in range(num_images):
        axes[i // 10, i % 10].imshow(images[i])
        axes[i // 10, i % 10].set_title( labels[i])
        axes[i // 10, i % 10].axis('off')
    plt.show()

# Display 20 training images
display_images(x_train, y_train, num_images=20)



# Step 4: Build the CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 5: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the Model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Step 7: Evaluate the Model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Step 6: Save model
model.save('drowsiness_model3.h5')

# Step 7: Make Predictions


# Assuming new_image is a new image you want to predict
