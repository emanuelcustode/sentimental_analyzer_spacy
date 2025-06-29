import spacy
from spacy.tokens import DocBin
from datasets import load_dataset

ds = load_dataset("Alienmaster/SB10k", split="train")  # 1.024 Trainingsbeispiele :contentReference[oaicite:2]{index=2}

sample = ds.shuffle(seed=42).select(range(3500))

nlp = spacy.blank("de")
doc_bin = DocBin()

label_map = {"positive": "POS", "neutral": "NEU", "negative": "NEG"}
for entry in sample:
    text = entry["Text"]
    label = entry["Sentiment"]
    cats = {lab: False for lab in label_map.values()}
    cats[label_map[label]] = True

    doc = nlp.make_doc(text)
    doc.cats = cats
    doc_bin.add(doc)

doc_bin.to_disk("training_data_set.spacy")