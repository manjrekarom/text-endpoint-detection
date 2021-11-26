"""
@Huy Anh Nguyen, 113094662
Created on Nov 18, 2021 (moved from jupyter notebook)
Last modified on Nov 20, 2021
"""
import numpy as np
import os
import time
import random
import torch
# Loading GloVe embedding

def load_glove(path="pretrain/glove.6B.300d.txt"):
    """
    Important: Keep the default name of Glove embedding
    It usually be glove.xB.yd.txt. We extract the number of tokens x
    and dimension y from that pattern.

    GloVe by default doesn't contain OOV (or UNK) word. So I preserved the 0 index for it.
    The embedding of <UNK> will be average of all other words.
    Note: there is unk in the vocab but it's not unknown token. So to distinguish, I used <UNK>
    """
    vector_dim = int(path.split(".")[2][:-1])
    print("Loading {}, vector dimension: {}".format(path, vector_dim))
    dtype = np.float32

    embedding = []
    word_to_idx = {"<UNK>": 0, "<PAD>": 1, "<ECON>": 2}
    idx_to_word = {0: "<UNK>", 1: "<PAD>", 2: "<ECON>"}
    start = time.time()
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            # Each line is word + embedding
            tmp = line.split()
            word_to_idx[tmp[0]] = i+3
            idx_to_word[i+3] = tmp[0]
            embedding.append(np.array(tmp[1:], dtype=dtype))

    embedding = np.array(embedding)
    unk_vector = embedding.mean(axis=0)
    pad_vector = np.random.choice(unk_vector, len(unk_vector), replace=False)
    econ_vector = np.random.choice(unk_vector, len(unk_vector), replace=False)

    embedding = np.vstack([unk_vector, pad_vector, econ_vector, embedding])
    print("Finished in {} second(s)".format(round(time.time() - start, 2)))
    return embedding, word_to_idx, idx_to_word


def generate_mask_fasttext(model, tokenizer, pad_vector, seq, max_sequence_length=32):
    """
    Generate mask for RNN models using fasttext
    Compatible with torch.utils.rnn.pack_padded_sequence
    """
    list_tokens = tokenizer.tokenize_batch(seq)
    mask = [min(len(x), max_sequence_length) for x in list_tokens]         
    res = []
    for tokens in list_tokens:
        if len(tokens) > max_sequence_length:
            tokens = tokens[:max_sequence_length]
            tmp = []
            for token in tokens:
                tmp.append(model.get_word_vector(token))  
            res.append(np.array(tmp))
        else:
            tmp = []
            for token in tokens:
                tmp.append(model.get_word_vector(token))  

            tmp += [pad_vector]*(max_sequence_length - len(tmp))
            res.append(np.array(tmp))
    return torch.tensor(np.array(res)), mask


def generate_model_path(model_type, embedding_type, check_dir = "checkpoints"):
    """
    Use for checking and generating path for checkpoint saving
    """
    cnt = 0
    while True:
        tmp = "{}_{}_{}".format(model_type, embedding_type, cnt)
        path = os.path.join(check_dir, tmp)
        if not os.path.isdir(path):
            break
        cnt += 1
    
    return path, tmp


def flat_accuracy(preds, labels):
    '''
    One-hot label accuracy
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    #labels_flat = np.argmax(labels, axis=1).flatten()
    return pred_flat == labels

def train_test_split(array, test_size, seed=0):
    """
    Manual train test split
    Use on M1 Mac (sklearn is not supported)
    """
    random.seed(seed)
    len_test = int(len(array)*test_size)
    len_train = len(array) - len_test
    idx_test = random.sample(list(range(len(array))), len_test)
    train = []
    test = []
    for i in range(len(array)):
        if i in idx_test:
            test.append(array[i])
        else:
            train.append(array[i])

    return train, test
