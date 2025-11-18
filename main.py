from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
import io, json


from model import RashCNN

app = FastAPI()

#Serve frontend 
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()

#Load class names
with open("class_names.json") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# Load model
model = RashCNN(num_classes)
model.load_state_dict(torch.load("best_rash_model.pth", map_location="cpu"))
model.eval()

#Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#Prediction Endpoint 
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_t = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        outputs = model(img_t)

        # Convert logits to probabilities
        probs = F.softmax(outputs, dim=1)[0]

        # Get top 3 class indices
        top3_prob, top3_idx = torch.topk(probs, 3)

    # Prepare response
    results = []
    for prob, idx in zip(top3_prob, top3_idx):
        results.append({
            "class": class_names[idx.item()],
            "probability": float(prob.item())  # convert tensor → float
        })

    return {
        "top3": results
    }
