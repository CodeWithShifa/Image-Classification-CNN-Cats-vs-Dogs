import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Faltu messages hide karne ke liye

import tensorflow as tf
from tensorflow.keras import layers, models, datasets

print("System Ready! ✅")

# 1. Data load karna (CIFAR-10 dataset)
print("Data load ho raha hai... Isme thoda waqt lag sakta hai.")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 2. Images ko normalize karna (0-1 range mein lana)
train_images, test_images = train_images / 255.0, test_images / 255.0

# 3. AI Model ka Structure (Dimagh) banana
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10) # 10 mukhtalif cheezon ko pehchanne ke liye
])

# 4. Model ko batana ke seekhna kaise hai
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 5. Training shuru karna
print("\nTraining shuru ho rahi hai... Computer images ko dekh kar seekh raha hai.")
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# 6. Model ko hamesha ke liye save karna
model.save('cat_dog_model.h5')

print("\n-------------------------------------------")
print("Mubarak ho! Training mukammal ho gayi. 🎉")
print("Aapka model 'cat_dog_model.h5' ke naam se save ho gaya hai.")
print("-------------------------------------------")