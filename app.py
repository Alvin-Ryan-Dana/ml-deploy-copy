import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

app = app = FastAPI(
        title="Animalink Model Deployment",
        description="Documentation of Model Apis for AnimaLink Corp",
        version="1.0",
        contact={
            "name": "Our Github Repository",
            "url": "https://github.com/AnimaLink",
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/AnimaLink/backend-api/blob/main/LICENSE",
        },
    )
model_path = 'AnimaLink2.h5'
model = keras.models.load_model(model_path)

animal_classes = {0 : 'Anjing ajag',  
       1 : 'Ikan koi',
       2 : 'Jalak bali', 
       3 : 'Merak biru',
       4 : 'Merak hijau',
       5 : 'Sanca bola',
       6 : 'Sanca hijau',
       7 : 'Shiba inu', 
       8 : 'Tenggiling',
       9 : 'Turkish angora'}

extinct_animals = {"Tenggiling", "Jalak bali", "Anjing ajag", "Merak hijau", "Sanca hijau"}

def preprocess_image(image):
    img = Image.open(image.file).convert("RGB")
    img = img.resize((150, 150)) 
    img_array = np.asarray(img)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_array = preprocess_image(file)
        predictions = model.predict(image_array)
        class_index = np.argmax(predictions[0])
        predicted_animal = animal_classes.get(class_index, "Unknown Animal")

        if predicted_animal in extinct_animals:
            status = "extinct"
            message = f"Warning! {predicted_animal} is an extinct animal. Selling them is prohibited."
        else:
            status = "not extinct"
            message = f"The predicted animal is {predicted_animal}. You can sell them."

        return JSONResponse(content={"predicted_animal": predicted_animal, "status": status, "message": message})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0',port=8080)
