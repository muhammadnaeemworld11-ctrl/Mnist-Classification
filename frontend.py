import streamlit as st 
import requests
import numpy as np
import tensorflow as tf
from PIL import Image

# Web app ka main  screen par dikhane ke liye
st.title("ANN Model with Streamlit")

# App ke baare mein choti si description
st.write("This is a number classification model.")

# Pehle se train kiye gaye ANN model (.h5 file) ko load karne ke liye
model = tf.keras.models.load_model("ANN_model.h5")

# 1. User se browser ke zariye image upload karwane ke liye widget (JPG, PNG, JPEG allowed hain)
Uploaded_image = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

# Check karne ke liye ke user ne image upload ki hai ya nahi (taake error na aaye)
if Uploaded_image is not None:
    # Image ko open karna aur classification ke liye usay GrayScale (Black & White) mein convert karna
    gray_img = Image.open(Uploaded_image).convert("L")
    
    # Upload ki gayi GrayScale image ko web screen par 150px width ke sath dikhana
    st.image(gray_img, caption="Uploaded GrayScale Image", width=150)

    # 3. Image ka size tabdeel karke 28x28 pixels karna kyunke model isi size par train hua hai
    resized_image = gray_img.resize((28, 28))
    
    # 4. Image ko numpy array mein badalna aur pixel values (0-255) ko normalize karke 0 aur 1 ke darmiyan lana
    img_array = np.array(resized_image).astype("float32") / 255.0
    
    # 5. 2D array (28x28) ko flatten karke 1D array (1, 784) mein badalna kyunke ANN ko single vector input chahiye
    img_flattened = img_array.reshape((1, 28 * 28))
    
    # 6. Prediction (Classification) ka process shuru karna
    # Jab user "Classification" ke button par click karega:
    if st.button("Classification"):
        # Flattened image ko model mein bhej kar probabilities (predictions) hasil karna
        prediction = model.predict(img_flattened)
        
        # Sab se zyada probability wali class ka index (number) nikalna
        predicted_class = int(np.argmax(prediction))
        
        # Final result yaani pehchana hua number screen par show karna
        st.write(f"Predicted Class: {predicted_class}")
