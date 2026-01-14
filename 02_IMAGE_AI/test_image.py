import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 1. Model load karein
# Warning ko handle karne ke liye compile=False rakha hai kyunke hum sirf predict kar rahe hain
model = tf.keras.models.load_model('cat_dog_model.h5', compile=False)

# 2. Image ka path dein (Check karein ke filename sahi ho)
img_path = 'my_pet.jpg' 

# CIFAR-10 ki 10 classes ki list
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

try:
    # Image ko preprocess karna
    img = image.load_img(img_path, target_size=(32, 32)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction karna
    prediction = model.predict(img_array)
    
    # Sabse bari value ka index nikalna
    result_index = np.argmax(prediction)
    result_name = classes[result_index]
    
    # Final Result Print karna
    print("\n" + "="*40)
    print(f"AI KA JAWAB: Yeh ek **{result_name.upper()}** hai! 🐾")
    print("="*40)
    
except Exception as e:
    print(f"Error: {e}")
    print(f"\nPhoto nahi mili! Check karein ke '{img_path}' folder mein majood hai.")