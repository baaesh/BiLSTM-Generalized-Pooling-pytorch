import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class NN4SNLI(nn.Module):

    def __init__(self, args, data):
        super(NN4SNLI, self).__init__()

        self.args = args

        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # fine-tune the word embedding
        self.word_emb.weight.requires_grad = False
        # <unk> vectors is randomly initialized
        nn.init.normal_(self.word_emb.weight.data[0])

        # character embedding
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=0)
        self.charCNN = CharCNN(args)

        # BiLSTM encoder with shortcut connections
        self.SeqEnc = SeqEncoder(args)

        # vector-based multi-head attention
        for i in range(args.num_heads):
            s2t = s2tSA(args)
            setattr(self, f's2tSA_{i}', s2t)

        # fully-connected layers for classification
        self.fc1 = nn.Linear(args.num_heads * 4 * 2 * args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.num_heads * 4 * 2 * args.hidden_dim + args.hidden_dim, args.hidden_dim)
        self.fc_out = nn.Linear(args.hidden_dim, args.class_size)
        self.relu = nn.ReLU()


    def get_s2tSA(self, i):
        return getattr(self, f's2tSA_{i}')


    def forward(self, batch):
        p, p_lengths = batch.premise
        h, h_lengths = batch.hypothesis

        # word embedding
        # (batch, seq_len, word_dim)
        p = self.word_emb(p)
        h = self.word_emb(h)

        # character embedding
        if not self.args.no_char_emb:
            # (batch, seq_len, max_word_len)
            char_p = batch.char_p
            char_h = batch.char_h
            batch_size, seq_len_p, _ = char_p.size()
            batch_size, seq_len_h, _ = char_h.size()

            # (batch * seq_len, max_word_len)
            char_p = char_p.view(-1, self.args.max_word_len)
            char_h = char_h.view(-1, self.args.max_word_len)

            # (batch * seq_len, max_word_len, char_dim)
            char_p = self.char_emb(char_p)
            char_h = self.char_emb(char_h)

            # (batch, seq_len, len(FILTER_SIZES) * num_feature_maps)
            char_p = self.charCNN(char_p).view(batch_size, seq_len_p, -1)
            char_h = self.charCNN(char_h).view(batch_size, seq_len_h, -1)

        p = torch.cat([p, char_p], dim=-1)
        h = torch.cat([h, char_h], dim=-1)

        # BiLSTM sequence encoder
        p = self.SeqEnc(p, p_lengths)
        h = self.SeqEnc(h, h_lengths)

        # vector-based multi-head attention
        v_ps = []
        v_hs = []
        for i in range(self.args.num_heads):
            s2tSA = self.get_s2tSA(i)
            v_p = s2tSA(p)
            v_h = s2tSA(h)
            v_ps.append(v_p)
            v_hs.append(v_h)

        v_p = torch.cat(v_ps, dim=-1)
        v_h = torch.cat(v_hs, dim=-1)

        v = torch.cat([v_p, v_h, (v_p - v_h).abs(), v_p * v_h], dim=-1)

        # fully-connected layers
        out = self.fc1(v)
        out = self.relu(out)
        out = self.fc2(torch.cat([v, out], dim=-1))
        out = self.relu(out)
        out = self.fc_out(out)

        return out


class SeqEncoder(nn.Module):

    def __init__(self, args):
        super(SeqEncoder, self).__init__()

        self.args = args
        self.emb_dim = args.word_dim + len(args.FILTER_SIZES) * args.num_feature_maps

        for i in range(args.num_layers):
            if i == 0:
                lstm_input_dim = self.emb_dim
            else:
                lstm_input_dim = self.emb_dim + 2 * args.hidden_dim
            lstm_layer = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=args.hidden_dim,
                bidirectional=True,
                batch_first=True
            )
            setattr(self, f'lstm_layer_{i}', lstm_layer)


    def get_lstm_layer(self, i):
        return getattr(self, f'lstm_layer_{i}')


    def forward(self, x, lengths):
        lens, indices = torch.sort(lengths, 0, True)

        x_sorted = x[indices]

        for i in range(self.args.num_layers):
            if i == 0:
                lstm_in = pack(x_sorted, lens.tolist(), batch_first=True)
            else:
                lstm_in = pack(torch.cat([x_sorted, lstm_out], dim=-1), lens.tolist(), batch_first=True)
            lstm_layer = self.get_lstm_layer(i)
            lstm_out, hid = lstm_layer(lstm_in)
            lstm_out = unpack(lstm_out, batch_first=True)[0]

        _, _indices = torch.sort(indices, 0)
        out = lstm_out[_indices]

        return out


class s2tSA(nn.Module):

    def __init__(self, args):
        super(s2tSA, self).__init__()

        self.fc1 = nn.Linear(2 * args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, 2 * args.hidden_dim)

        self.relu = nn.ReLU()


    def forward(self, x):
        # (batch, seq_len, word_dim)
        f = self.relu(self.fc1(x))
        f = F.softmax(self.fc2(f), dim=-2)

        # (batch, word_dim)
        s = torch.sum(f * x, dim=-2)

        return s


class CharCNN(nn.Module):

    def __init__(self, args):
        super(CharCNN, self).__init__()

        self.args = args
        self.FILTER_SIZES = args.FILTER_SIZES

        for filter_size in args.FILTER_SIZES:
            conv = nn.Conv1d(1, args.num_feature_maps, args.char_dim * filter_size, stride=args.char_dim)
            setattr(self, 'conv_' + str(filter_size), conv)


    def forward(self, x):
        batch_seq_len, max_word_len, char_dim = x.size()

        # (batch * seq_len, 1, max_word_len * char_dim)
        x = x.view(batch_seq_len, 1, -1)

        conv_result = [
            F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(x)), max_word_len - filter_size + 1).view(-1,
                                                                                                                    self.args.num_feature_maps)
            for filter_size in self.FILTER_SIZES]

        out = torch.cat(conv_result, 1)

        return out