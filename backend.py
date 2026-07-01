from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Model load
model = tf.keras.models.load_model("ANN_model.h5")

# First route
@app.get("/")
def Hello():
    return {"message": "Welcome to my ANN model's API"}

# Second route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read bytes data
    image_bytes = await file.read()
    
    # 2. Wrap bytes in BytesIO and open with PIL
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    # 3. Resize image to 28x28 if required by your model
    img = img.resize((28, 28))
    
    # 4. Convert to numpy array and normalize
    img_array = np.array(img).astype("float32") / 255.0
    
    # 5. Reshape for ANN input (1, 784)
    img_flattened = img_array.reshape((1, 28 * 28))
    
    # 6. Predict
    prediction = model.predict(img_flattened)
    predicted_class = int(np.argmax(prediction))
    
    return {"predicted_class": predicted_class}
