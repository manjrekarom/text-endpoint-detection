"""
@Huy Anh Nguyen, 113094662
Created on Nov 20, 2021
Last modified on Nov 20, 2021
"""
import argparse
import random
import os

from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from dataloader import SentenceDataset
from embeddings.glove import GloveModel
from embeddings.tokenizer import SpacyTokenizer
from utils import load_glove, flat_accuracy, generate_mask_fasttext, generate_model_path

from preprocess import preprocess


parser = argparse.ArgumentParser(description="Inference Endpoint")
parser.add_argument('--data', type=str, required=False, default="dataset/sentiment/eval.txt", help="Path to data file")
parser.add_argument('--model_path', type=str, required=False, default="checkpoints/GRU_glove_0/model.tar", help="Path to model checkpoint")
parser.add_argument('--batch-size', type=int, default=64, help="batch size")
# parser.add_argument('--test', action='store_false')
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

# Load dataset
if args.test:
    with open(args.data, "r") as f:
        sentences = f.readlines()
        data = [preprocess(sen) for sen in sentences]
    dataset = SentenceDataset(data) 

else:
    with open(args.data, "r") as f:
        lines = f.readlines()
        data = []
        label = []
        for line in lines:
            # line in form of sentence, label
            tmp = line.replace("\n","").split("\t")
            data.append(preprocess(tmp[0]))
            label.append(int(tmp[1]))
    dataset = SentenceDataset(data, label)

test_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)

checkpoint = torch.load(args.model_path)

if checkpoint['embedding'] == "glove":
    model = GloveModel(checkpoint['num_class'], torch.tensor(np.zeros(checkpoint['embedding_size']), dtype=torch.float32), model=checkpoint['model'], max_sequence_length=checkpoint['max_seq_length'])

elif checkpoint['embedding'] == "fasttext":
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
tokenizer = checkpoint['tokenizer']

res = []
total_accuracy = None
for step, batch in enumerate(tqdm(test_loader)):
    model.eval()
    if args.test:
        # Save to file
        data = tokenizer(batch).to(device)
        pred = model(data)
        logits = nn.functional.softmax(pred, dim=1).detach().cpu().numpy()
        res += np.argmax(logits, axis=1).tolist()
        # for sen, label in zip(batch, np.argmax(logits, axis=1).tolist()):
        #     res.append(sen + "," + str(label))

    else:
        data = tokenizer(batch[0]).to(device)
        pred = nn.functional.softmax(model(data), dim=1).detach().cpu().numpy()
        label = batch[1].cpu().numpy()
        if total_accuracy is None:
            total_accuracy = flat_accuracy(pred, label)
        else:
            total_accuracy = np.concatenate([total_accuracy, flat_accuracy(pred, label)])          

# Report result
if args.test:
    tmp = os.path.split(args.data)
    out_file_name = "result_" + tmp[1]
    out_path = os.path.join(tmp[0], out_file_name)
    with open(out_path, "w") as f:
        for line, label in zip(sentences, res):
            item = line.replace("\n", "") + "\t" + str(label)
            f.writelines("{}\n".format(item))

    print("Saved result file in {}".format(out_path))

else:
    print("Evaluation accuracy {}".format(round(100*total_accuracy.sum()/len(total_accuracy), 2)))


