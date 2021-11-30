"""
@Huy Anh Nguyen, 113094662
Created on Nov 29, 2021 (moved from jupyter notebook)
Last modified on Nov 29, 2021
"""
import argparse
import random
import os

from tqdm import trange, tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
# from sklearn.model_selection import train_test_split
from utils import train_test_split

from dataloader import SentenceDataset
# from embeddings.glove import GloveModel
# from embeddings.fasttext import FastTextModel
# from embeddings.BERT_embedding import BERTEmbModel
#from embeddings.BERT_embedding import

from seq_model.Att import Encoder, AttDecoder, AttSeq2Seq
from embeddings.tokenizer import SpacyTokenizer
from utils import load_glove, flat_accuracy, generate_mask_fasttext, generate_model_path

from preprocess import preprocess

parser = argparse.ArgumentParser(description="Train Detection Endpoint")
parser.add_argument('--source', '--pos', type=str, required=False, default="dataset/endpoint/seq_source.txt", help="path to source sequences dataset")
parser.add_argument('--target', '--neg', type=str, required=False, default="dataset/endpoint/seq_target.txt", help='path to target sequences dataset')
parser.add_argument("--ratio", type=float, default=0.2, help="Train test split ratio")
parser.add_argument('--batch-size', '--batch', type=int, default=64, help="batch size")
parser.add_argument('--epoch', type=int, default=20, help="Number of training epoch(s)")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--glove", type=str, default = "pretrain/glove.6B.300d.txt")

args = parser.parse_args()

EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
MAX_SEQ = 64
HIDDEN_DIM = 300

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

with open(args.source, "r") as f:
    sentences = f.readlines()
    source = [preprocess(sen) for sen in sentences]


target_vocab = {"<WORD>": 0, # normal words
                "<SSEN>": 1, # start of sentence
                "<ECON>": 2, # end of context
                "<PAD>" : 3, # padding token
                }
with open(args.target, "r") as f:
    target = f.readlines()

train_source, test_source = train_test_split(source, test_size = args.ratio)
train_target, test_target = train_test_split(target, test_size = args.ratio)

train_dataset = SentenceDataset(train_source, train_target)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

test_dataset = SentenceDataset(test_source, test_target)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=True)

def process_target(target, target_vocab=target_vocab):
    res = []
    for sen in target:
        tokens = sen.replace("\n", "").split()
        tmp = [target_vocab[token] for token in tokens]
        if len(tmp) < MAX_SEQ:
            tmp += [target_vocab["<PAD>"]] * (MAX_SEQ - len(tmp))
        else:
            tmp = tmp[:MAX_SEQ]
        res.append(tmp)
    return torch.tensor(res)

embedding_matrix, word_to_idx, idx_to_wor = load_glove(args.glove)
tokenizer = SpacyTokenizer(word_to_idx = word_to_idx, max_sequence_length=MAX_SEQ)

encoder = Encoder(torch.tensor(embedding_matrix), HIDDEN_DIM)
decoder = AttDecoder(target_vocab, encoder.get_hidden_dim(), embedding_matrix.shape[1], HIDDEN_DIM)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttSeq2Seq(encoder, decoder, device)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
criterion = nn.CrossEntropyLoss().to(device)

best_val_acc = 0
best_val_epoch = 1
# writer = SummaryWriter(os.path.join('runs', tb_path))
writer = SummaryWriter()

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    total_train_accuracy = None

    # Keep track of LR during scheduler testing
    writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'])

    with tqdm(train_loader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch):
            tepoch.set_description("Epoch {}/{}".format(epoch+1, EPOCHS))
            data = tokenizer(batch[0]).to(device)
            label = process_target(batch[1]).to(device)
            pred = model(data, label, True)
            print("DEBUG")

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            logits = nn.functional.softmax(pred.detach().cpu(), dim=1).numpy()
            label_ids = batch[1].cpu().numpy()
            if total_train_accuracy is None:
                total_train_accuracy = flat_accuracy(logits, label_ids)
            else:
                total_train_accuracy = np.concatenate([total_train_accuracy, flat_accuracy(logits, label_ids)])

            curr_loss = total_train_loss/(step+1)
            writer.add_scalar("Training loss", curr_loss)
            tepoch.set_postfix(loss = curr_loss)

    train_accuracy = total_train_accuracy.sum()/len(total_train_accuracy)
    writer.add_scalar("Training accuracy", train_accuracy)

    print("Train accuracy: {}, loss: {}".format(round(100*train_accuracy, 2), curr_loss))

    # ====== Validation ==========
    total_val_loss = 0
    total_val_accuracy = None
    model.eval()
    for step, batch in enumerate(eval_loader):
        label = batch[1].to(device)
        if args.embedding == "glove":
            data = tokenizer(batch[0]).to(device)
            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, label)
        elif args.embedding == "fasttext":
            data, mask = generate_mask_fasttext(ft, tokenizer, pad_vector, batch[0])
            with torch.no_grad():
                data = data.to(device)
                pred = model(data, mask)
        elif args.embedding == "bert":
            with torch.no_grad():
                if args.model == "bert":
                    data = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt").to(device)
                    pred = model(data)
                else:
                    data = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")['input_ids'].to(device)
                    pred = model(data)            

        total_val_loss += loss.item()
        logits = nn.functional.softmax(pred, dim=1).detach().cpu().numpy()
        label_ids = batch[1].cpu().numpy()
        if total_val_accuracy is None:
            total_val_accuracy = flat_accuracy(logits, label_ids)
        else:
            total_val_accuracy = np.concatenate([total_val_accuracy, flat_accuracy(logits, label_ids)])     
        writer.add_scalar("Validation loss", total_val_loss/(step+1))
    val_accuracy = total_val_accuracy.sum()/len(total_val_accuracy)
    print("Validation accuracy: {}, loss: {}".format(round(100*val_accuracy, 2), total_val_loss/(step+1)))
    writer.add_scalar("Validation accuracy", val_accuracy)
    
    # Track val loss each epoch on Scheduler
    scheduler.step(total_val_loss/(step+1))

    # ====== Save model ==========
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_epoch = step + 1
        print("Found a better model. Saving ...")
        torch.save({'epoch': epoch,
                    'num_class': NUM_CLASS,
                    'max_seq_length': MAX_SEQ,
                    'model': args.model,
                    'embedding_size': embedding_size,
                    'embedding': args.embedding,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'tokenizer': tokenizer,
                    }, os.path.join(save_path,'best.tar'))

    torch.save({'epoch': epoch,
                'num_class': NUM_CLASS,
                'max_seq_length': MAX_SEQ,
                'model': args.model,
                'embedding_size': embedding_size,
                'embedding': args.embedding,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'tokenizer': tokenizer,
                }, os.path.join(save_path,'model.tar'))

    print()

print("Best val accuracy {} on epoch {}".format(best_val_acc, best_val_epoch))



