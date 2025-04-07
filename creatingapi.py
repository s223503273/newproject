from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io

app = FastAPI()

# Load your model
model = torch.load("best.pt")
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess the image
    # (resize, normalize, convert to tensor, etc.)
    
    # Make prediction
    with torch.no_grad():
        result = model(image)  # Adjust based on your model
    
    return {"prediction": str(result)}
