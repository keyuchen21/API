#################
# initial testing
#################

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def home():
#     return {"message": "BERT API is running!"}

#################
# V2
#################

# from fastapi import FastAPI
# from pydantic import BaseModel
# from transformers import pipeline

# # Initialize FastAPI app
# app = FastAPI()

# # Load BERT sentiment analysis model
# sentiment_model = pipeline("sentiment-analysis")

# # Define input data model
# class TextInput(BaseModel):
#     text: str

# @app.get("/")
# def home():
#     return {"message": "BERT API is running!"}

# @app.post("/predict")
# def predict(input: TextInput):
#     result = sentiment_model(input.text)
#     return {"input": input.text, "result": result}


#################
# V3
#################


from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F



class TextInput(BaseModel):
    text: str

app = FastAPI()

# --- Startup: load once, warm up ---
MODEL_PATH = "./bert_model"  # 👈 use local folder instead of Hugging Face ID
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()  # inference mode
torch.set_grad_enabled(False)
# Optional CPU threading cap (tune!): torch.set_num_threads(1)

LABELS = model.config.id2label


@app.get("/")
def home():
    return {"message": "BERT API is running!"}

@app.post("/predict")
def predict(inp: TextInput):
    encoded = tokenizer(inp.text, return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        logits = model(**encoded).logits
        probs = F.softmax(logits, dim=-1)[0]
    top = int(torch.argmax(probs).item())
    return {"label": LABELS[top], "score": float(probs[top])}
