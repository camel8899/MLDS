
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset,DataLoader

import os
import numpy as np

import json


# In[2]:


VOCAB_SIZE = 6772
BOS_TOKEN = 0
EOS_TOKEN = 1
hidden_size = 256 
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


# In[4]:


class S2VT(nn.Module):
    def __init__(self, ninp, nhid, ntoken, nlayers, bsz, dropout):
        super(S2VT, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.ntoken = ntoken
        self.nlayers = nlayers
        self.dropout = dropout
        self.bsz = bsz
        self.padding = torch.zeros(bsz, nhid)
        self.pad = torch.zeros(bsz, ninp)
        
        self.lstm1 = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.lstm2 = nn.LSTM(nhid + nhid, nhid, nlayers, dropout=dropout)
        
        self.hidden2out = nn.Linear(nhid, ntoken)
        self.vocab2emb = nn.Linear(ntoken, nhid)
        
        self.softmax = nn.LogSoftmax()
        
    def init_hidden(self, bsz):
        return (Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda(),
                Variable(torch.zeros(self.nlayers, bsz, self.nhid)).cuda())
    
    def forward(self, inp, hid1, hid2, target_len):
        
        # Encoding stage
        
        out1, hid1 = self.lstm1(inp, hid1)
        padding = Variable(self.padding.repeat(len(inp), 1).view(len(inp), 1, -1)).cuda()
        out2, hid2 = self.lstm2(torch.cat((padding, out1), 2), hid2)        
      
        # Decoding stage

        padding = Variable(self.pad.repeat(target_len,1).view(target_len,self.bsz,-1)).cuda()
        
        bos_onehot = Variable(torch.FloatTensor([1] + [0 for _ in range(self.ntoken-1)])).cuda()
        out2_vocab_emb = self.vocab2emb(bos_onehot).view(1, self.bsz, -1)
        
        out1, hid1 = self.lstm1(padding, hid1)
        
        output2 = Variable(torch.zeros(target_len, self.bsz, self.ntoken)).cuda()
        
        hid2_t = hid2
        
        for i in range(target_len):
            out2_t, hid2_t = self.lstm2(torch.cat((out2_vocab_emb, out1[i].view(1, self.bsz, -1)), 2), hid2_t)
            out2_t = self.softmax(self.hidden2out(out2_t.view(out2_t.size(0), -1)))
            output2[i] = out2_t
            output2_vocab_emb = self.vocab2emb(out2_t.view(1, self.bsz, -1))
        
        return output2


# In[4]:

def train():
    epoch = 25
    bsz = 1
    vocabs = Vocab('MLDS_hw2_data/training_label.json')
    DS = TA_Dataset('MLDS_hw2_data/training_data/feat', 'MLDS_hw2_data/training_label.json')
    model = S2VT(4096, 256, 6772, 1, 1, 0.5).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
   
    # In[5]:
    
    
    def repackage_hidden(h):
        '''Wraps hidden states in new Variables to detach them from history'''
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)
    
    
    # In[6]:
    
    
    from tqdm import tqdm
    hid1, hid2 = model.init_hidden(1), model.init_hidden(1)
    for i in range(epoch):
        total_loss = 0
        for data in tqdm(DS):
            optimizer.zero_grad()
            feat = data[0]
            feat = Variable(torch.from_numpy(feat).view(feat.shape[0], 1, -1).float()).cuda()
            caption = data[1][0]
            
            for ch in '.!()':
                if ch in caption:
                    caption = caption.replace(ch, '')
            print(caption)            
            tar = Variable(torch.LongTensor([vocabs.vocab2index[word] for word in caption.split()]+[EOS_TOKEN])).cuda()
            output = model(feat, hid1, hid2, len(tar))
            
            output = output.view(-1, 6772)
            loss = criterion(output, tar)
            total_loss += loss.data[0]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()
            hid1, hid2 = repackage_hidden(hid1), repackage_hidden(hid2)
        print('Epoch {} Loss: {}'.format(i, total_loss/len(DS)))
        
    with open('output_model.ffs', 'wb') as f:
        torch.save(model, f)
train()
