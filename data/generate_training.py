import os
import json
import argparse

from jsonpath_rw import jsonpath, parse
from ftfy import fix_text

def load_acts(dataroot):
  act_file = os.path.join(dataroot, 'dialog_acts.json')
  if not os.path.exists(act_file):
      print(f'File doesn\'t exist as {act_file}')
      exit(1)
  with open(act_file) as f:
      acts = json.load(f)
  return acts


def load_dialogues(dataroot, split, domains=None, max_dialogues=None):
  dialogue_root = os.path.join(dataroot, split)
  print(f'Finding dialogue files in {dialogue_root}')
  dialogue_files = os.listdir(os.path.join(dataroot, split))
  print(f'Found {len(dialogue_files)} files. First file is {dialogue_files[0]}')
  dialogues = []
  for df in dialogue_files:
    filepath = os.path.join(dialogue_root, df)
    print(f'File {filepath}')
    print('=' * 30)
    with open(filepath, 'r+') as f:
      print(f'Reading text...')
      text = fix_text(f.read())
      try:
        print('Parsing json...')
        dialogues_json = json.loads(text)
        for dialogue in dialogues_json:
          if domains:
            if set(dialogue['services']).intersection(set(domains)):
              dialogues.append(dialogue)
              if max_dialogues and len(dialogues) > max_dialogues:
                return dialogues
          else:
            dialogues.append(dialogue)
            if max_dialogues and len(dialogues) > max_dialogues:
              print(f'Total {len(dialogues)} dialogues.')
              return dialogues
      except:
        print(f'Couldn\'t process {df}.')
  print(f'Total {len(dialogues)} dialogues.')
  return dialogues


def load_utt_slot(dialogue, speaker='USER'):
  utt_slot = []
  for dialogue in dialogues:
    turns = dialogue['turns']
    for turn in turns:
      # print('Processing turn...')
      if turn['speaker'].lower() != speaker.lower():
        # print(f'Skipping because not {speaker}')
        continue
      slots = []
      for frame in turn['frames']:
        # print('Processing frame...')
        state = frame['state']
        if state['active_intent'].lower() != 'none':
          for key, value in state['slot_values'].items():
            slots.extend(value)
          break
      utterance = turn['utterance']
      utt_slot.append((utterance, slots))
  return utt_slot


def split_sentences_at_slots(utt_slot, save_path):
  positives, negatives = [], []
  for utt, slots in utt_slot:
    utt = utt.lower().strip()
    positives.append(utt)
    for slot in slots:
      idx = utt.find(slot)
      if idx != -1:
        neg = utt[:idx]
        if len(neg) > 0:
          negatives.append(neg)
  if save_path:
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    pos_folder = os.path.join(save_path, 'positives.txt')
    neg_folder = os.path.join(save_path, 'negatives.txt')
    print(f'Saving results at {pos_folder} and {neg_folder}')
    with open(pos_folder, 'w+') as f:
      f.writelines(line + '\n' for line in positives)
    with open(neg_folder, 'w+') as f:
      f.writelines(line + '\n' for line in negatives)
  return positives, negatives
      

if __name__ == "__main__":    
  parser = argparse.ArgumentParser(description='Generate training examples for MultiWoz 2.2')
  parser.add_argument('--dataroot', type=str, required=True)
  parser.add_argument('--split', type=str, required=True)
  parser.add_argument('--max_dialogues', type=int, default=1e6)
  parser.add_argument('--domains', nargs='+', default=[], help='List of acceptable domains of dialogue.')
  parser.add_argument('--out_path', type=str, default='./')

  args = parser.parse_args()
  print(f'Generating data. Returning conversations belonging only to domain of {args.domains}')
  dialogues = load_dialogues(args.dataroot, args.split, max_dialogues=args.max_dialogues, \
  domains=args.domains)
  print('First dialogue', dialogues[1])
  
  utt_slot = load_utt_slot(dialogues)
  print(utt_slot)

  pos, neg = split_sentences_at_slots(utt_slot, args.out_path)
  print(f'Generated {len(pos)} positive and {len(neg)} negative examples')
