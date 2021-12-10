"""
@Huy Anh Nguyen, 113094662
Created on Dec 10, 2021 (after project submission, just for demo/showcase)
Last modified on Dec 10, 2021
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from dataloader import SentenceDataset
from embeddings.glove import GloveModel
from embeddings.fasttext import FastTextModel
from embeddings.BERT_embedding import BERTEmbModel
from embeddings.tokenizer import SpacyTokenizer
from utils import load_glove, generate_mask_fasttext

parser = argparse.ArgumentParser(description="Demo Train Detection Endpoint")
parser.add_argument('--checkpoint', type=str, required=False, default="checkpoints/test/model.tar", help="path to checkpoint")
parser.add_argument("--glove", type=str, default = "pretrain/glove.6B.300d.txt")

args = parser.parse_args()
device = torch.device('cpu')

checkpoint = torch.load(args.checkpoint, map_location=device)
MAX_SEQ = checkpoint['max_seq_length']
NUM_CLASS = checkpoint['num_class']

embedding_size = None
if checkpoint["embedding"] == "glove":
    embedding_matrix, word_to_idx, idx_to_wor = load_glove(args.glove)
    tokenizer = SpacyTokenizer(word_to_idx = word_to_idx, max_sequence_length=MAX_SEQ)
    model = GloveModel(NUM_CLASS, torch.tensor(embedding_matrix), model=checkpoint["model"])
    embedding_size = list(model.embedding.weight.shape)
elif checkpoint["embedding"] == "fasttext":
    import fasttext
    import fasttext.util

    ft = fasttext.load_model('pretrain/cc.en.300.bin')
    pad_vector = ft.get_input_matrix().mean(axis=0)
    tokenizer = SpacyTokenizer(max_sequence_length=MAX_SEQ)
    model = FastTextModel(NUM_CLASS, model=checkpoint["model"])
elif checkpoint["embedding"] == "bert":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=MAX_SEQ, additional_special_tokens = ["<ECON>"])
    if checkpoint["model"] == "bert":
        # Pure BERT
        from clf_model.BERT import BERTModel
        model = BERTModel(num_class=NUM_CLASS)
    else:
        # BERT Embedding with other models
        model = BERTEmbModel(NUM_CLASS, model=checkpoint["model"])

# Load saved checkpoint
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print()
print("="*40)
print("Start chatting...")
print()
store_input = ""
while True:
    user_input = input("User: ")

    if user_input == "<END>":
        print("Machine: See you later!")
        break

    if store_input == "":
        store_input = user_input
    
    else:
        store_input = store_input + " " + user_input

    with torch.no_grad():
        if checkpoint["embedding"] == "glove":
            data = tokenizer(store_input).to(device)
            pred = model(data)
        elif checkpoint["embedding"] == "fasttext":
            data, mask = generate_mask_fasttext(ft, tokenizer, pad_vector, store_input, max_sequence_length=MAX_SEQ)
            data = data.to(device)
            pred = model(data, mask)
        elif checkpoint["embedding"] == "bert":
            if checkpoint["model"] == "bert":
                data = tokenizer(store_input, padding=True, truncation=True, return_tensors="pt").to(device)
                pred = model(data)
            else:
                data = tokenizer(store_input, padding=True, truncation=True, return_tensors="pt")['input_ids'].to(device)
                pred = model(data)

    logits = np.argmax(nn.functional.softmax(pred, dim=1).numpy())
    if logits:
        print("Machine: Response!!!")
        store_input = "" # Reset
