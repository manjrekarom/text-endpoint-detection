"""
@Huy Anh Nguyen, 113094662
Created on Nov 19, 2021 (moved from jupyter notebook)
Last modified on Nov 20, 2021
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
from embeddings.glove import GloveModel
from embeddings.fasttext import FastTextModel
from embeddings.BERT_embedding import BERTEmbModel
#from embeddings.BERT_embedding import
from embeddings.tokenizer import SpacyTokenizer
from utils import load_glove, flat_accuracy, generate_mask_fasttext, generate_model_path

from preprocess import preprocess

parser = argparse.ArgumentParser(description="Train Detection Endpoint")
parser.add_argument('--pos-data', '--pos', type=str, required=False, default="dataset/sentiment/pos.txt", help="path to positive dataset")
parser.add_argument('--neg-data', '--neg', type=str, required=False, default="dataset/sentiment/neg.txt", help='path to negative dataset')
parser.add_argument("--ratio", type=float, default=0.2, help="Train test split ratio")
parser.add_argument('--embedding', type=str, required=False, default="bert", help="choose embedding to model: glove, fasttext, BERT")
parser.add_argument('--model', type=str, required=False, default="GRU", help="choose model to train: GRU, LSTM, CNN, BERT")
parser.add_argument('--batch-size', '--batch', type=int, default=64, help="batch size")
parser.add_argument('--epoch', type=int, default=20, help="Number of training epoch(s)")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--glove", type=str, default = "pretrain/glove.6B.300d.txt")

args = parser.parse_args()

EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
NUM_CLASS = 2
MAX_SEQ = 64

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Load data
with open(args.pos_data, "r") as f:
    sentences = f.readlines()
    pos_data = [preprocess(sen) for sen in sentences]

with open(args.neg_data, "r") as f:
    sentences = f.readlines()
    neg_data = [preprocess(sen) for sen in sentences]

train_pos, eval_pos = train_test_split(pos_data, test_size = args.ratio)
train_neg, eval_neg = train_test_split(neg_data, test_size = args.ratio)

train_data = train_pos + train_neg
train_label = [1]*len(train_pos) + [0]*len(train_neg)

eval_data = eval_pos + eval_neg
eval_label = [1]*len(eval_pos) + [0]*len(eval_neg)

# Create dataloader
train_dataset = SentenceDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

eval_dataset = SentenceDataset(eval_data, eval_label)
eval_loader = DataLoader(eval_dataset, batch_size = BATCH_SIZE, shuffle=True)

# Create directory for model checkpoints
save_path, tb_path = generate_model_path(args.model, args.embedding)
os.makedirs(save_path)
print("Checkpoints will be saved into {}".format(save_path))

embedding_size = None
if args.embedding == "glove":
    embedding_matrix, word_to_idx, idx_to_wor = load_glove(args.glove)
    tokenizer = SpacyTokenizer(word_to_idx = word_to_idx, max_sequence_length=MAX_SEQ)
    model = GloveModel(NUM_CLASS, torch.tensor(embedding_matrix), model=args.model)
    embedding_size = list(model.embedding.weight.shape)
elif args.embedding == "fasttext":
    import fasttext
    import fasttext.util

    ft = fasttext.load_model('pretrain/cc.en.300.bin')
    pad_vector = ft.get_input_matrix().mean(axis=0)
    tokenizer = SpacyTokenizer(max_sequence_length=MAX_SEQ)
    model = FastTextModel(NUM_CLASS, model=args.model)
elif args.embedding == "bert":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', model_max_length=MAX_SEQ, additional_special_tokens = ["<ECON>"])
    if args.model == "bert":
        # Pure BERT
        from model.BERT import BERTModel
        model = BERTModel(num_class=NUM_CLASS)
    else:
        # BERT Embedding with other models
        model = BERTEmbModel(NUM_CLASS, model=args.model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

            label = batch[1].to(device)

            if args.embedding == "glove":
                data = tokenizer(batch[0]).to(device)
                pred = model(data)
            elif args.embedding == "fasttext":
                data, mask = generate_mask_fasttext(ft, tokenizer, pad_vector, batch[0], max_sequence_length=MAX_SEQ)
                data = data.to(device)
                pred = model(data, mask)
            elif args.embedding == "bert":
                if args.model == "bert":
                    data = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt").to(device)
                    pred = model(data)
                else:
                    data = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt")['input_ids'].to(device)
                    pred = model(data)

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
            tepoch.set_postfix(loss = curr_loss)

    train_accuracy = total_train_accuracy.sum()/len(total_train_accuracy)
    writer.add_scalar("Training accuracy", train_accuracy, epoch)
    writer.add_scalar("Training loss", curr_loss, epoch)

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
    writer.add_scalar("Training loss", total_val_loss/(step+1), epoch)
    writer.add_scalar("Validation accuracy", val_accuracy, epoch)
    
    # Track val loss each epoch on Scheduler
    scheduler.step(total_val_loss/(step+1))

    # ====== Save model ==========
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_val_epoch = epoch + 1
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
