import os
import json
import random
import argparse

from jsonpath_rw import jsonpath, parse
from ftfy import fix_text

def load_sentences(filepath):
  with open(filepath, 'r+') as f:
    return f.readlines()

def add_econ(sentences):
  return ''.join([sent.rstrip() + ' <ECON> ' for sent in sentences])

def random_pool(pos_sentences):
  tfm_pos_sentences = []
  i = 0
  while i < len(pos_sentences):
    r = random.randint(1, 10)
    if r <= 5:
      tfm_pos_sentences.append(add_econ([pos_sentences[i]]))
      i += 1
    elif r <= 7:
      sentences = pos_sentences[i:i+2]
      tfm_pos_sentences.append(add_econ(sentences))
      i += 2
    elif r <= 9:
      sentences = pos_sentences[i:i+3]
      tfm_pos_sentences.append(add_econ(sentences))
      i += 3
    elif r <= 10:
      sentences = pos_sentences[i:i+4]
      tfm_pos_sentences.append(add_econ(sentences))
      i += 4
  return tfm_pos_sentences

def convert_to_tokens(utterances):
  tokenized_utterances = []
  for utts in utterances:
    utts = utts.replace("\n", " ")
    utt_arr = utts.split("<ECON>")
    # print('len of utt_arr before', utt_arr)
    utt_arr = [utt.strip().replace(".", "").replace(",", "").lower() for utt in utt_arr]
    utt_arr = list(filter(lambda x: len(x.rstrip()) > 0, utt_arr))
    # print('len of utt_arr after', utt_arr)
    tokenized_utt = []
    for idx, utt in enumerate(utt_arr):
      tokens = utt.split(" ")
      if idx == 0:
        tokenized_utt += ["<SSEN>"] + ["<WORD>"] * (len(tokens)-2) + ["<ECON>"]
      else:
        tokenized_utt += ["<WORD>"] * (len(tokens) - 1) + ["<ECON>"]
    tokenized_utterances.append(" ".join(tokenized_utt))
  return tokenized_utterances

def convert_negative_to_tokens(utterances):
  tokenized_utterances = []
  for utt in utterances:
    utt = utt.replace("\n", " ")
    # print(utts)
    # print(utt_arr)
    assert len(utt.split("<ECON>")) < 2, "Negative sentences should not have <ECON>"
    tokens = utt.split(" ")
    tokenized_utt = ["<SSEN>"] + ["<WORD>"] * (len(tokens)-1)
    tokenized_utterances.append(" ".join(tokenized_utt))
  return tokenized_utterances

def randomize(source, target):
  combined = list(zip(source, target))
  random.shuffle(combined)
  source[:], target[:] = zip(*combined)
  return source, target

def transform_negative(neg_sentences):
  tfmed_neg_sentences = []
  for utt in neg_sentences:
    utt = utt.replace("\n", " ")
    utt = utt.replace(".", "").replace(",", "").rstrip().lower()
    tfmed_neg_sentences.append(utt)
  return tfmed_neg_sentences

def save_data(source, source_file, target, target_file):
  with open(source_file, 'w') as f:
    f.writelines(line + '\n' for line in source)
  with open(target_file, 'w') as f:
    f.writelines(line + '\n' for line in target)

def remove_econ(sentences):
  tfmed_sentences = []
  for sent in sentences:
    tfmed_sentences.append(sent.replace('<ECON>', '').rstrip())
  return tfmed_sentences


if __name__ == "__main__":    
  parser = argparse.ArgumentParser(description='Generate training examples for MultiWoz 2.2')
  parser.add_argument('pos_path', type=str)
  parser.add_argument('neg_path', type=str)
  parser.add_argument('--max_length', type=int, default=100, \
  help='Max length of sentences.')
  parser.add_argument('--out_path', type=str, default='./')

  args = parser.parse_args()
  print(f'Transforming data for seq2seq from {args.pos_path} and {args.neg_path}')
  
  pos = load_sentences(args.pos_path)
  neg = load_sentences(args.neg_path)
  print('Displaying first few positives and negatives\n')
  print('Positives')
  print('=' * 10)
  [print(line[:-1]) for line in pos[:5]]
  
  print('\nNegatives')
  print('=' * 10)
  [print(line[:-1]) for line in neg[:5]]
  
  tfmed_pos = random_pool(pos)
  print('Displaying few transformed positives and negatives\n')
  print('Transformed Positives')
  print('=' * 10)
  [print(line, len(line.split('<ECON>'))-1, len(line.rstrip().split(" "))) for line in tfmed_pos[:5]]
  print('')
  print('Tokenized Positives')
  print('-' * 10)
  tokenized_utts_pos = convert_to_tokens(tfmed_pos)
  [print(line, len(line.rstrip().split(" "))) for line in tokenized_utts_pos[:5]]
  print('')

  tfmed_neg = transform_negative(neg)
  print('Transformed Negatives')
  print('=' * 10)
  [print(line, len(line.rstrip().split(" "))) for line in tfmed_neg[:5]]
  print('')
  print('Tokenized Negatives')
  print('-' * 10)
  tokenized_utts_neg = convert_negative_to_tokens(tfmed_neg)
  [print(line, len(line.rstrip().split(" "))) for line in tokenized_utts_neg[:5]]
  print('')

  source = remove_econ(tfmed_pos) + remove_econ(tfmed_neg)
  target = tokenized_utts_pos + tokenized_utts_neg
  source, target = randomize(source, target)

  source_file = os.path.join(args.out_path, 'source.txt')
  target_file = os.path.join(args.out_path, 'target.txt')
  print(f'Saving source and target at {source_file} and {target_file}')
  save_data(source, source_file, target, target_file)
