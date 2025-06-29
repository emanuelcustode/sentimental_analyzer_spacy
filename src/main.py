import spacy
from fastapi import FastAPI
from pydantic import BaseModel


nlp = spacy.load("../output/model-best")

app = FastAPI(title="Sentiment Analysis")

class TextInput(BaseModel):
    text: str

@app.post("/sentiment")
def sentiment(text: TextInput):
    doc = nlp(text.text)
    cats = doc.cats
    label = max(cats, key=cats.get)


    return {"text": text.text,
            "prediction": label,
            "scores": {k: round(v, 4) for k, v in cats.items()}
            }