import cv2
import requests
import base64
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from fastapi import FastAPI
from bson.json_util import dumps
import json
import threading
import time

# --- MongoDB Atlas Setup ---
uri = "mongodb+srv://vaibhava:Thanos123@cluster0.dhvti8r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["face_recognition"]
collection = db["predictions"]

# --- FastAPI App ---
app = FastAPI()

@app.get("/latest_predictions")
def get_latest_predictions():
    data = collection.find().sort("timestamp", -1).limit(10)
    return json.loads(dumps(data))


# --- Function to Send Image to Model API and Save to MongoDB ---
def process_image_and_store(image_path):
    try:
        image = cv2.imread(image_path)
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = base64.b64encode(buffer).decode('utf-8')

        # Send to your model API
        api_url = "http://127.0.0.1:8000/process_frame"
        payload = {"image": image_bytes}
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            prediction = response.json()
            prediction["timestamp"] = datetime.utcnow()

            # Optional: Flatten bounding box if needed
            if "bounding_box" in prediction:
                box = prediction["bounding_box"]
                prediction.update({
                    "x": box.get("x"),
                    "y": box.get("y"),
                    "width": box.get("width"),
                    "height": box.get("height")
                })
                del prediction["bounding_box"]

            collection.insert_one(prediction)
            print("✅ Stored prediction in MongoDB!")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print("Response:", response.text)

    except Exception as e:
        print("❌ Error processing image:", str(e))


# --- Background Runner for One Image ---
def background_worker():
    image_path = "face detection.jpg"  # <<< Replace with actual image
    while True:
        process_image_and_store(image_path)
        time.sleep(5)  # wait 5 seconds and re-run (you can adjust or remove this if just once)


# --- Start background processing on script run ---
if __name__ == "__main__":
    threading.Thread(target=background_worker, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
