import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch import optim
import os
import json
import numpy as np
from tqdm import tqdm
import random

VOCAB_SIZE = 3003 
DATA_DIR = './MLDS_hw2_data/training_data/feat'
TEST_DIR = './MLDS_hw2_data/testing_data/feat'
SAVE_DIR = './save'
LABEL_PATH = './MLDS_hw2_data/training_label.json'
ID_PATH = './MLDS_hw2_data/testing_id.txt'
USE_CUDA = torch.cuda.is_available()
BOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2
hidden_size =256 
model = 'S2VT'
postfix = 'eva'
class Vocab:
	def __init__(self,label_path):
		print("Building Vocab")
		with open(label_path,'r') as f:
			self.label = json.load(f)
		self.word2index = {'<BOS>':0, '<EOS>':1, '<UNK>':2}
		self.index2word = {0:'<BOS>',1:'<EOS>',2:'<UNK>'}
		self.word2count = {'<BOS>':1,'<EOS>':1,'<UNK>':1}
		self.num_words = 3
		self.build()
	def build(self):
		for l in self.label:
			for line in l["caption"]:
				for ch in '.!()':
					if ch in line:
						line = line.replace(ch,'')
				for w in line.strip().split():
					if w not in self.word2count.keys():
						self.word2count[w] = 1
					else:
						self.word2count[w] += 1
		sorted_word = [w for (w,c) in sorted(self.word2count.items(), key = lambda x: x[1], reverse = True)]
		for w in sorted_word[:3000]:
			self.word2index[w] = self.num_words
			self.index2word[self.num_words] = w
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

class Testset(Dataset):
	def __init__(self,data_dir,id_path):
		print("Preparing dataset")
		self.data_dir = data_dir
		self.label = []
		with open(id_path,'r') as f:
                    for line in f:
                        self.label.append(line.strip())
	def __len__(self):
		return len(self.label)
	
	def __getitem__(self,index):
		avi_id = self.label[index]+'.npy'
		data = np.load(os.path.join(self.data_dir,avi_id))
		return data,self.label[index] 

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
    
    def forward(self, input_feat, hidden1, hidden2, target, tf_ratio = 0):
        target_len = len(target)
 
        output1, hidden1 = self.lstm1(input_feat, hidden1)
        padding = Variable(self.padding1.repeat(len(input_feat), 1).view(len(input_feat), 1, -1)).cuda()
        output2, hidden2 = self.lstm2(torch.cat((padding, output1), 2), hidden2)        
      

        pad = Variable(self.padding2.repeat(target_len,1).view(target_len,self.batch_size,-1)).cuda()
        
        bos_onehot = Variable(torch.FloatTensor([1] + [0 for _ in range(VOCAB_SIZE-1)])).cuda()
        output2_vocab_emb = self.vocab2emb(bos_onehot).view(1, self.batch_size, -1)
        
        output1, hidden1 = self.lstm1(pad, hidden1)
        
        output2 = Variable(torch.zeros(target_len, self.batch_size, VOCAB_SIZE)).cuda()
        
        hidden2_t = hidden2
        
        for i in range(target_len):
            output2_t, hidden2_t = self.lstm2(torch.cat((output2_vocab_emb, output1[i].view(1, self.batch_size, -1)), 2), hidden2_t)
            output2_t = self.softmax(self.hidden2out(output2_t.view(output2_t.size(0), -1)))
            output2[i] = output2_t
            if random.uniform(0,1) <= tf_ratio:
                oh = [0 for _ in range(VOCAB_SIZE)]
                oh[target[i]] = 1
                word_onehot = Variable(torch.FloatTensor(oh)).cuda()
                output2_vocab_emb = self.vocab2emb(word_onehot).view(1,self.batch_size,-1)
            else:         
                output2_vocab_emb = self.vocab2emb(output2_t).view(1, self.batch_size, -1)
        
        return output2

if USE_CUDA:
	print("USE CUDA")
else:
	print("Suggest using cuda")

BATCH_SIZE = 1
V = Vocab(LABEL_PATH)
DS = Testset(TEST_DIR,ID_PATH)
DS = TA_Dataset(DATA_DIR,LABEL_PATH)
CHECK_PATH = './save/S2VT/256-clip25_drop3_lr-3_nt_3000_ss/119.tar'
checkpoint = torch.load(CHECK_PATH)
M = S2VT(4096,hidden_size,1,1,0.3)
M.load_state_dict(checkpoint['model'])
M.eval()
h1 = M.init_hidden(1)
h2 = M.init_hidden(1)
MAX_LENGTH = 20
if USE_CUDA:
	M = M.cuda()

def output_sen(index_list,V):
	output = ""
	for i in index_list:
		if i == EOS_TOKEN:
			#output += "."
			break
		elif i == BOS_TOKEN:
			continue
		else:
			output += V.index2word[i]+" "
	return output	

with open('output.csv','w') as f:
	for data in DS:
		#caption_id = data[1]
		caption = random.choice(data[1])
		data = data[0]
		for ch in '.!()':
			if ch in caption:
				caption = caption.replace(ch, '')
		target_seq = []
		for word in caption.split():
			if word in V.word2index.keys():
				target_seq.append(V.word2index[word])
		else:
			target_seq.append(UNK_TOKEN)
			target_seq.append(EOS_TOKEN)
        
		feat = Variable(torch.from_numpy(data).view(data.shape[0],1,-1).float())
		feat = feat.cuda() if USE_CUDA else feat
		output = M(feat,h1,h2,target_seq)
		output = output.view(-1,VOCAB_SIZE)
		_,index = torch.max(output,1)
		print(output_sen(index.data.tolist(),V))
		#f.write('{},{}\n'.format(caption_id,output_sen(index.data.tolist(),V)))
