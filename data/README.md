# Data for Text Endpoint Detection #

We are using MultiWOZ multi-domain conversational dataset. MultiWOZ contains annotations
for intent detection, slot filling, dialogue state tracking and dialogue policy tasks.

One problem of splitting at random spans is we might accidentally generate false negative examples. For example, for the sentence:
```
I'm looking for an expensive restaurant in the center of the city. 
```

If we split this sentence randomly, it might be split as: 
```
I'm looking for an expensive restaurant.
```

Thus we leverage information in the slots for generating true negative examples by cutting the sentences before the slots. In the above example the slot occurs at expensive and center giving us two negative examples:
```
1. I'm looking for an 
2. I'm looking for an expensive restaurant in the 
```

Copy the multiwoz dataset from https://github.com/budzianowski/multiwoz.

To generate positive and negative examples from multiwoz dataset use following command:
```
python generate_training.py \
--dataroot ./multiwoz/data/MultiWOZ_2.2/ \
--split train \
--domains restaurant hotel \
--max_dialogues 5000
```
This generates two files positives.txt and negatives.txt in the `--out_path` consisting
of positive and negative sentences. Train the model with this dataset using `clf_train.py`.


To transform positive and negative examples for sequence labeling use following command:
```
python transform_seq2seq.py positives.txt negatives.txt 
```
This will generate two files source.txt and target.txt. Train the model with this dataset
using `seq_train.py`
