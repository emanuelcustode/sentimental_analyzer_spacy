import spacy
from spacy_tokens import DocBin
import json
import argparse
from datasets import load_datasets



def convert_json_to_spacy(lang="de"):

    ds = load_datasets("amazon_reviews_multi", "de", split="train")
    sample = ds.shuffle(seed=42).select(range(1000))

    nlp = spacy.blank(lang)
    doc_bin = DocBin()


    for entry in sample:
        text = entry["review_body"]
        label_rating = entry["stars"]
        if label_rating <= 2:
            cats = {"NEG": True, "NEU": False, "POS": False}
        if label_rating == 3:
            cats = {"NEG": False, "NEU": True, "POS": False}
        else:
            cats = {"NEG": False, "NEU": False, "POS": True}
        doc = nlp.make_doc(text)
        doc.cats = cats
        doc_bin.add(doc)

    doc_bin.to_disk("training_data_set.spacy")
