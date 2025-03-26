from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form, File, UploadFile
from typing import Annotated


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def hello():
    return { "message": "hello" }


@app.post("/train")
async def train(
    weights: Annotated[UploadFile, File()],
    step: Annotated[int, Form()]
):
    with open("received.pth", "wb") as f:
        content = await weights.read()
        f.write(content)
    return {
        "status": "train",
        "metrics": {
            "step": step,
            "accuracy": step**2 / 100,
            "loss": (200 - step) / 100.
        }
    }
