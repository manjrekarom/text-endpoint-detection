# Endpoint Detection Project

## CSE538 NLP, Fall 2018

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
