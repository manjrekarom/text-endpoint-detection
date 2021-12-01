# Endpoint Detection Project

## CSE538 NLP, Fall 2021

### Anh, Omkar, Sriram

## Task 1: Using classification

- Requirements

Download and install fastText

```bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
python setup.py install
```

Make a directory called **pretrain/** in the root. Download glove and fastText inside it.

Download Glove

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

Download fastText pretrained embedding

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz
```

Verify that glove* files and cc.en.300.bin should be inside **pretrain/**


Install huggingface Transformers

```bash
pip install transformers
```

- Training the model

There are two formulations, endpoint detection as a classification, and as sequence labeling.

```bash
# seq labeling
python seq_train.py --source dataset/endpoint/source.txt --target dataset/sentiment/target.txt --model bert
```

- Results on UCI Sentiment Analysis Task

| Embedding | Model | Best val acc | On epoch | Running time |
| :-------: | :---: | :----------: | :------: | :----------: |
|   GloVe   |  GRU  |    0.845     |    10    |     1.67     |
|           | LSTM  |    0.853     |    10    |     1.63     |
|           |  CNN  |    0.833     |    10    |     1.59     |
| fastText  |  GRU  |    0.831     |    10    |     1.73     |
|           | LSTM  |    0.815     |    10    |     1.64     |
|           |  CNN  |    0.683     |    10    |     2.40     |
|   BERT    |  GRU  |    0.855     |    10    |    16.87     |
|           | LSTM  |     0.86     |    10    |    16.32     |
|           |  CNN  |    0.882     |    10    |    24.16     |
|           | BERT  |    0.953     |    10    |     3.23     |
