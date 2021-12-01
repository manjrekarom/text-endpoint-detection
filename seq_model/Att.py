"""
@Huy Anh Nguyen, 113094662
Created on Nov 29, 2021 
Last modified on Nov 29, 2021

=============================
Implementation of Attention + Seq2Seq model with Teacher Forcing

Idea is based on "Sequence Labeling Approach to the Task of Sentence Boundary Detection" by Le et al
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_matrix, enc_hidden_dim, dec_hidden_dim, num_layers=1,max_sequence_length=64):
        super().__init__()
        # using GloVe embedding
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        embedding_dim = embedding_matrix.shape[-1]
        self.GRUs = nn.GRU(embedding_dim, enc_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        self.hidden_dim = (int(self.GRUs.bidirectional)+1)*num_layers*enc_hidden_dim

        self.fc_hidden = nn.Linear(self.hidden_dim, dec_hidden_dim)

    def forward(self, sequence):
        """
        Inputs:
            sequence:   [batch_size, max_sequence_length] 
                        contains indices from source vocabulary 
        
        Output:
            outputs:    [batch_size, max_sequence_length, 2*hidden_dim]
                        last layer hidden representation of each token in the sequence
            hidden: [num_layers, batch_size, hidden_dim] 
                    hidden state of every token in sequence input
            cell:   [num_layers, batch_size, hidden_dim]
                    last cell state of RNN
        """
        emb = self.embedding(sequence)
        outputs, hidden = self.GRUs(emb)
        
        # Return hidden of every token
        # And cell for input cell of Decoder (also a stacked LSTM)
        hidden = self.fc_hidden(torch.flatten(torch.swapaxes(hidden, 0, 1), 1, 2))

        return outputs, hidden.unsqueeze(0)

    def get_hidden_dim(self):
        return self.hidden_dim

class AttentionLayer(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()
        self.att = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.att_score = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs):
        """
        Inputs: dec_hidden [num_layers, batch_size, hidden_dim]
                enc_outputs [batch_size, max_sequence_length, 2*hiddem_dim]

        Outputs: score of each token in the input sequence
        """
        max_sequence_length = enc_outputs.shape[1]

        dec_hidden = torch.swapaxes(dec_hidden, 0, 1) # [batch_size, num_layers, hidden_dim]
        dec_hidden = torch.flatten(dec_hidden, 1, 2).unsqueeze(dim=1) # [batch_size, 1, num_layers*hidden_dim]
        dec_hidden = torch.repeat_interleave(dec_hidden, max_sequence_length, dim=1)
        x = torch.cat([dec_hidden, enc_outputs], dim=-1) # [batch_size, max_sequence_length, num_layers*hidden_dim + hidden_dim]
        x = torch.tanh(self.att(x)) # [batch_size, max_sequence_length, dec_hidden_dim]
        att_scores = self.att_score(x).squeeze(-1) # [batch_size, max_sequence_length]

        return F.softmax(att_scores, dim=1) # [batch_size, max_sequence_length]

class AttDecoder(nn.Module):
    """
    A step of Decoder
    Run multiple time in decoding process
    """
    def __init__(self, word2idx, enc_hidden_dim, embedding_dim, dec_hidden_dim, num_layers=1, max_sequence_length=64):
        super().__init__()
        # Embedding Layer
        #self.word2idx = word2idx # Dictionary of index to word
        self.embedding = nn.Embedding(len(word2idx), embedding_dim)

        # RNN Layer
        self.GRUs = nn.GRU(embedding_dim + enc_hidden_dim, dec_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False) # Only 1 direction

        # Attention Layer
        self.att_layer = AttentionLayer(enc_hidden_dim, dec_hidden_dim)

        # Output layer
        self.output = nn.Linear(dec_hidden_dim, len(word2idx))

    # def forward(self, input_tokens, hidden, cell, enc_outputs):
    def forward(self, input_tokens, hidden, enc_outputs):
        """
        Inputs: 
            input: [batch_size] a word of target vocabulary 
            hidden: hidden state of every tokens from encoder
            enc_outputs: [batch_size, max_sequence_length, 2*hidden_dim]
                    cell state of encoder -> feed into initial cell state of decoder

        Output:
            Predicted logits of [batch_size, target_vocab_size]
        """
        input_tokens = input_tokens.unsqueeze(-1) # [batch_size, 1]
        emb = self.embedding(input_tokens) # [batch_size, 1, embedding_dim]
        att_scores = self.att_layer(hidden, enc_outputs).unsqueeze(1) # [batch_size, 1, max_seq_length]
        # print(att_scores.shape)

        hidden_input = torch.bmm(att_scores, enc_outputs) # [batch_size, 1, 2*hidden_dim]
        rnn_input = torch.cat([emb, hidden_input], dim=-1) # [batch_size, 1, 2*hidden_dim + emb_dim]

        output, hidden = self.GRUs(rnn_input, hidden)
        #print(output.shape, output.squeeze(1).shape)
        
        pred = self.output(output.squeeze(1))

        return pred, hidden

class AttSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, max_sequence_length=64):
        super().__init__()
        # Target words only contain <ECON> and <WORD>
        # self.target_word2idx = {"<WORD>": 0, # normal words
        #                         "<SSEN>": 1, # start of sentence
        #                         "<ECON>": 2, # end of context
        #                         "<PAD>" : 3, # padding token
        #                         }

        # self.target_idx2word = {0: "<WORD>",
        #                         1: "<SSEN>",
        #                         2: "<ECON>",
        #                         3: "<PAD>",
        #                         }

        self.max_sequence_length = max_sequence_length
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def forward(self, input_seq, target_seq, teacher_forcing_ratio=0.5, training = True):
        """
        Teacher forcing will be disable during inference
        """
        batch_size = input_seq.shape[0]
        dec_vocab_size = self.decoder.embedding.weight.shape[0]
        res = torch.zeros((batch_size, self.max_sequence_length, dec_vocab_size)).to(self.device)

        #enc_outputs, enc_hidden, enc_cell = self.encoder(input_seq)
        enc_outputs, enc_hidden = self.encoder(input_seq)

        dec_inputs = torch.ones((batch_size,), dtype=torch.int64).to(self.device) # First input to decoder

        for i in range(self.max_sequence_length):
            pred, dec_hidden = self.decoder(dec_inputs, enc_hidden, enc_outputs)
            res[:, i, :] = pred
            #print("DEBUG", pred.shape)
            if random.random() > teacher_forcing_ratio and training:
                dec_inputs = target_seq[:, min(i+1, self.max_sequence_length-1)]
                #print("DEBUG", dec_inputs.shape)
            else:
                dec_inputs = torch.argmax(F.softmax(pred, dim=1), dim=1)
                # if len(dec_inputs.shape) == 0:
                #     # When batch has only 1 example
                #     dec_inputs = dec_inputs.unsqueeze(0)
                # #print("DEBUG", dec_inputs.shape)
        return res







