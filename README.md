# Endpoint Detection Project

## CSE538 NLP, Fall 2021

### Members:
* Huy Anh Nguyen (113094662)
* Omkar Manjrekar (113287703)
* Sriram Vithala (113166835)

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

Verify that glove* files and cc.en.300.bin are inside **pretrain/**


Install huggingface Transformers

```bash
pip install transformers
```

- Training the model

There are two formulations, endpoint detection as a classification, and as sequence labeling.

```bash
# seq labeling
python clf_train.py --model BERT --pos-data dataset/endpoint/positives_all_domains.txt --neg-data negatives_all_domain.txt
```

## Task 2: Sequential Labelling

```bash
# seq labeling
python seq_train.py --source dataset/endpoint/source.txt --target dataset/sentiment/target.txt --model bert

python seq_train.py --source dataset/endpoint/source.txt --target dataset/sentiment/target.txt --model bert
```
