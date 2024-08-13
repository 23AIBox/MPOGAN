import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel


class Generator(nn.Module):
    '''MPOGAN Generator'''

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 vocab_size,
                 max_seq_len,
                 device:str,
                 oracle_init=False):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        if oracle_init:
            for p in self.parameters():
                nn.init.normal_(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        return h.to(self.device)

    def forward(self, inp, hidden):
        emb = self.embeddings(inp)
        emb = emb.view(1, -1, self.embedding_dim)
        out, hidden = self.gru(emb, hidden)
        out = self.gru2out(out.view(-1, self.hidden_dim))
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        samples = torch.zeros(num_samples,
                              self.max_seq_len).type(torch.LongTensor)
        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter] * num_samples))

        samples = samples.to(self.device)
        inp = inp.to(self.device)

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

        for i in range(num_samples):
            if samples[i][0] == 21:
                samples[i] = self.sample(1, start_letter=0)
                
        return samples

    def batchNLLLoss(self, inp, target):
        '''Negative Log-Likelihood Loss'''
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)

        loss = 0

        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        return loss

    def batchPGLoss(self, inp, target, reward):
        '''Policy Gradient Loss'''
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(inp[i], h)
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]] * reward[j]

        loss /= batch_size
        loss = torch.as_tensor(loss, dtype=torch.float64)

        return loss


class Discriminator(nn.Module):
    '''MPOGAN Discriminator and LLM-AAP'''

    def __init__(self, device:str, max_len=27, dropout=0.1):
        super().__init__()
        self.model_checkpoint = "./models/esm2_t6_8M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.esm_model = EsmModel.from_pretrained(self.model_checkpoint)
        self.device = device
        self.max_len = max_len
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.get_seq_representations(x)
        x = self.dropout(x)
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.output_layer(x)
        return torch.softmax(x, dim=1)

    def batchClassify(self, x):
        outputs = self.forward(x)
        outputs_target = torch.argmax(outputs, dim=1)
        return outputs_target
    
    def get_seq_representations(self, seqs):
        inputs = self.tokenizer(seqs,
                        return_tensors="pt",
                        max_length=self.max_len,
                        padding="max_length",
                        truncation=True)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.esm_model(**inputs)

        attention_mask =inputs["attention_mask"]

        last_hidden_states = []
        for i in range(len(outputs[0])):
            last_hidden_states.append(outputs[0][i][attention_mask[i] == 1][1:-1])

        seq_representations = []
        for i in range(len(last_hidden_states)):
            seq_representations.append(torch.mean(last_hidden_states[i], dim=0))
        seq_representations = torch.stack(seq_representations)

        return seq_representations
