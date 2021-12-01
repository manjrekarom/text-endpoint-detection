# Endpoint Detection Project

## CSE538 NLP, Fall 2021

### Anh, Omkar, Sriram

## Task 1: Using classification

- Requirements

Download Glove

```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

Download and install fastText

```bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
python setup.py install
```

Download fastText pretrained embedding

```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz
```

After that, put these files into **/pretrain/**

Install huggingface Transformers

```bash
pip install transformers
```

- Training the model

```bash
python train.py --pos-data dataset/sentiment/pos.txt --neg-data dataaset/sentiment/neg.txt --embedding glove --model gru
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
