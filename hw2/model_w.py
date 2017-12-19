import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch import optim
import os
import numpy as np
import json
from tqdm import tqdm

VOCAB_SIZE = 6772
DATA_DIR = './MLDS_hw2_data/training_data/feat'
SAVE_DIR = './save'
LABEL_PATH = './MLDS_hw2_data/training_label.json'
USE_CUDA = torch.cuda.is_available()
BOS_TOKEN = 0
EOS_TOKEN = 1
hidden_size = 256 
postfix = 'clip_drop'

class Vocab:
	def __init__(self,label_path):
		print("Building Vocab")
		with open(label_path,'r') as f:
			self.label = json.load(f)
		self.vocab2index = {'<BOS>':0, '<EOS>':1}
		self.index2vocab = {0:'<BOS>',1:'<EOS>'}
		self.num_words = 2
		self.build()

	def build(self):
		for l in self.label:
			for line in l["caption"]:
				line = line.replace('.','')
				line = line.replace('!','')
				line = line.replace('(','')
				line = line.replace(')','')
				for w in line.strip().split():
					if w not in self.vocab2index.keys():
						self.vocab2index[w] = self.num_words
						self.index2vocab[self.num_words] = w
						self.num_words += 1			

class TA_Dataset(Dataset):
	def __init__(self,data_dir,label_path):
		print("Preparing dataset")
		self.data_dir = data_dir
		with open(label_path,'r') as f:
			self.label = json.load(f)

	def __len__(self):
		return len(self.label)
	
	def __getitem__(self,index):
		avi_id = self.label[index]["id"]+'.npy'
		data = np.load(os.path.join(self.data_dir,avi_id))
		return data,self.label[index]["caption"] 

class S2VT(nn.Module):
    def __init__(self, input_size, hidden_size, layer, batch_size, dropout):
        super(S2VT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = layer
        self.dropout = dropout
        self.batch_size = batch_size
        self.padding1 = torch.zeros(batch_size, hidden_size)
        self.padding2 = torch.zeros(batch_size, input_size)
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, layer, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size + hidden_size, hidden_size, layer, dropout=dropout)
        
        self.hidden2out = nn.Linear(hidden_size, VOCAB_SIZE)
        self.vocab2emb = nn.Linear(VOCAB_SIZE, hidden_size)
        
        self.softmax = nn.LogSoftmax()
        
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.layer, batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(self.layer, batch_size, self.hidden_size)).cuda())
    
    def forward(self, input_feat, hidden1, hidden2, target_len):
        
        
        out1, hidden1 = self.lstm1(input_feat, hidden1)
        padding = Variable(self.padding1.repeat(len(input_feat), 1).view(len(input_feat), 1, -1)).cuda()
        out2, hidden2 = self.lstm2(torch.cat((padding, out1), 2), hidden2)        
      

        pad = Variable(self.padding2.repeat(target_len,1).view(target_len,self.batch_size,-1)).cuda()
        
        bos_onehot = Variable(torch.FloatTensor([1] + [0 for _ in range(VOCAB_SIZE-1)])).cuda()
        out2_vocab_emb = self.vocab2emb(bos_onehot).view(1, self.batch_size, -1)
        
        out1, hidden1 = self.lstm1(pad, hidden1)
        
        output2 = Variable(torch.zeros(target_len, self.batch_size, VOCAB_SIZE)).cuda()
        
        hidden2_t = hidden2
        
        for i in range(target_len):
            out2_t, hidden2_t = self.lstm2(torch.cat((out2_vocab_emb, out1[i].view(1, self.batch_size, -1)), 2), hidden2_t)
            out2_t = self.softmax(self.hidden2out(out2_t.view(out2_t.size(0), -1)))
            output2[i] = out2_t
            output2_vocab_emb = self.vocab2emb(out2_t.view(1, self.batch_size, -1))
        
        return output2
 
def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)
    
ITER = 25
V = Vocab(LABEL_PATH)
DS = TA_Dataset(DATA_DIR,LABEL_PATH)
M = S2VT(4096, hidden_size, 1, 1, 0.5).cuda()
criterion = nn.NLLLoss().cuda()
optimizer = optim.Adam(M.parameters(), lr=1e-3)
       
h1 = M.init_hidden(1)
h2 = M.init_hidden(1)
for i in range(ITER):
    total_loss = 0
    for data in tqdm(DS):
        optimizer.zero_grad()
        feat = data[0]
        feat = Variable(torch.from_numpy(feat).view(feat.shape[0], 1, -1).float()).cuda()
        caption = data[1][0]
            
        for ch in '.!()':
            if ch in caption:
                caption = caption.replace(ch, '')
        tar = Variable(torch.LongTensor([V.vocab2index[word] for word in caption.split()]+[EOS_TOKEN])).cuda()
        output = M(feat, h1, h2, len(tar))
        
        output = output.view(-1, VOCAB_SIZE)
        loss = criterion(output, tar)
        total_loss += loss.data[0]
        loss.backward()
            
        torch.nn.utils.clip_grad_norm(M.parameters(), 0.25)
        optimizer.step()
        h1, h2 = repackage_hidden(h1), repackage_hidden(h2)
    print('Epoch {} Loss: {}'.format(i, total_loss/len(DS)))
        
