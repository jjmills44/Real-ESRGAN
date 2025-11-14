from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from py_real_esrgan.model import RealESRGAN
from PIL import Image
import torch
import io

app = FastAPI()

# ---- Load model once at startup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RealESRGAN(device, scale=4)
# This will download weights to ./weights if they are not present
model.load_weights("weights/RealESRGAN_x4.pth", download=True)

@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    # Read uploaded image into PIL
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Run Real-ESRGAN
    sr_img = model.predict(img)

    # Return PNG bytes
    buf = io.BytesIO()
    sr_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
