import os
import json
import argparse

from jsonpath_rw import jsonpath, parse


def load_acts(dataroot):
    act_file = os.path.join(dataroot, 'dialog_acts.json')
    if not os.path.exists(act_file):
        print(f'File doesn\'t exist as {act_file}')
        exit(1)
    with open(act_file) as f:
        acts = json.load(f)
    return acts


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Explore MultiWoz 2.2')
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--max_files', type=int, default=2)
    parser.add_argument('--out_path', type=str, default='./dialogues.json')

    args = parser.parse_args()
    acts = load_acts(args.dataroot)
    files = acts.keys()
    print('Number of Files found as keys in dialog acts:', len(files))
    print('')
    train_directory = os.path.join(args.dataroot, 'train')
    train_files = os.listdir(train_directory)
    jsonpath_expr = parse('$..utterance')
    utterances = []
    for idx, fpath in enumerate(train_files):
      if idx + 1 == args.max_files:
        break
      with open(os.path.join(train_directory, fpath)) as json_file:
        dialogue_dict = json.load(json_file)
        for match in jsonpath_expr.find(dialogue_dict):
          utterances.append(match.value)
    
    print(f'Found {len(utterances)} in {idx+1} files')
    with open(args.out_path, 'w+') as f:
      f.writelines(list(map(lambda x: x + '\n', utterances)))
    print(f'Utterances saved at {args.out_path}!')
