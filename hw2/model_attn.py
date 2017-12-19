import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
import json
import random
from tqdm import tqdm

VOCAB_SIZE = 3004 
DATA_DIR = './MLDS_hw2_data/training_data/feat'
SAVE_DIR = './save'
LABEL_PATH = './MLDS_hw2_data/training_label.json'
USE_CUDA = torch.cuda.is_available()
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
hidden_size = 256 
postfix = 'clip25_drop3_lr-3_3000_attn'

class Vocab:
	def __init__(self,label_path):
		print("Building Vocab")
		with open(label_path,'r') as f:
			self.label = json.load(f)
		self.word2index = {'<PAD>':0,'<BOS>':1, '<EOS>':2, '<UNK>':3}
		self.index2word = {0:'<PAD>',1:'<BOS>',2:'<EOS>',3:'<UNK>'}
		self.word2count = {'<PAD>':1,'<BOS>':1,'<EOS>':1,'<UNK>':1}
		self.num_words = 4 
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

def cap2index(caption,V):
	for ch in '.!()':
		if ch in caption:
			caption = caption.replace(ch, '')
	caption_index = [PAD_TOKEN for _ in range(50)]
	for i,word in enumerate(caption.split()):
		if word in V.word2index.keys():
			caption_index[i] = V.word2index[word]
		else:
			caption_index[i] = UNK_TOKEN
	caption_index[len(caption.split())] = EOS_TOKEN
	return caption_index

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
		data = torch.from_numpy(np.load(os.path.join(self.data_dir,avi_id))).float().cuda()
		caption_index = torch.LongTensor(cap2index(random.choice(self.label[index]["caption"]),V)).cuda()
		return data,caption_index

class Encoder(nn.Module):
	def __init__(self,input_size,hidden_size,layer=1):
		super(Encoder,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.layer = layer
		self.lstm = nn.LSTM(input_size,hidden_size,layer,batch_first = True)
	
	def forward(self,data,hidden):
		for i in range(self.layer):
			output,hidden = self.lstm(data,hidden)
		return output,hidden

	def init_hidden(self,batch_size):
		return Variable(torch.zeros(1,batch_size,self.hidden_size).cuda()),Variable(torch.zeros(1,batch_size,self.hidden_size).cuda())

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax()
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len).cuda()) # B x S

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
		
                attn_energies[b, i] = self.score(hidden[0][0][b], encoder_outputs[b,i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return self.softmax(attn_energies).unsqueeze(1)
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class Decoder(nn.Module):
	def __init__(self,input_size,hidden_size,layer=1):
		super(Decoder,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.layer = layer
		self.lstm = nn.LSTM(input_size,hidden_size,layer,batch_first = True)
		self.hidden2out = nn.Linear(hidden_size,VOCAB_SIZE)
		self.softmax = nn.LogSoftmax()
		self.embedding = nn.Embedding(VOCAB_SIZE,hidden_size)

	def forward(self,data,hidden):
		for i in range(self.layer):
			output,hidden = self.lstm(data,hidden)
			result = self.softmax(self.hidden2out(output).view(-1,VOCAB_SIZE))
			embed = self.embedding(torch.max(result,1)[1]).view(-1,1,hidden_size)
		return result, embed, hidden

	def init_hidden(self,batch_size):
		return Variable(torch.zeros(1,batch_size,self.hidden_size).cuda()),Variable(torch.zeros(1,batch_size,self.hidden_size).cuda())

ITER = 200
V = Vocab(LABEL_PATH)
DS = TA_Dataset(DATA_DIR,LABEL_PATH)
DL = DataLoader(DS,batch_size = 16)
print("Finish building dataloader")
E = Encoder(4096, hidden_size, 1).cuda()
D = Decoder(3*hidden_size, hidden_size, 1).cuda()
#A = Attn('general',hidden_size).cuda()
criterion = nn.NLLLoss().cuda()
#a_optim = optim.Adam(A.parameters(), lr=1e-3)
e_optim = optim.Adam(E.parameters(), lr=1e-3)
d_optim = optim.Adam(D.parameters(), lr=1e-3)
SS_RATIO = 0.5

directory = os.path.join(SAVE_DIR,  'S2VT_EnDe', '{}-{}'.format(hidden_size,postfix))
if not os.path.exists(directory):
    os.makedirs(directory)

SAVE = 2	   
for epoch in range(1,ITER):
	total_loss = 0
	for data,caption_index in tqdm(DL):
		e_optim.zero_grad()
#		a_optim.zero_grad()
		d_optim.zero_grad()

		feat = Variable(data.view(data.size()[0], 80, -1))
		target = Variable(caption_index)
		## stage 1 ##

		decoder_padding = Variable(torch.zeros(data.size()[0],80,2*hidden_size).cuda())
		encoder_hidden = E.init_hidden(data.size()[0])
		decoder_hidden = D.init_hidden(data.size()[0])
		
		encoder_output1, encoder_hidden = E(feat, encoder_hidden)
		decoder_result, decoder_embed, decoder_hidden = D(torch.cat((decoder_padding,encoder_output1),2),decoder_hidden)
	
		embed = Variable(torch.zeros(data.size()[0],1,hidden_size).cuda())
		## stage 2 ##
		loss = 0
		encoder_padding = Variable(torch.zeros(data.size()[0],caption_index.size()[1],4096).cuda())
		encoder_output2, encoder_hidden = E(encoder_padding,encoder_hidden)
		for i in range(caption_index.size()[1]):
			attn_energies = Variable(torch.zeros(data.size()[0], 80).cuda()) # B x S
			for b in range(data.size()[0]):
				for l in range(80):
					attn_energies[b,l] = torch.dot(decoder_hidden[0][0,b],encoder_output1[b,l])
			attn =	F.softmax(attn_energies).view(data.size()[0],1,-1)
			context = torch.bmm(attn,encoder_output1)
			decoder_input = torch.cat((embed,context,encoder_output2[:,i,:].unsqueeze(1)),2)
			decoder_result2, embed, decoder_hidden = D(decoder_input,decoder_hidden)
			'''
			if random.uniform(0,1) < SS_RATIO:
				embed = D.embedding(target[:,i]).view(-1,1,hidden_size)
			'''
			loss += criterion(decoder_result2,target[:,i])	
		
		total_loss += loss.data[0]
		loss.backward()
		torch.nn.utils.clip_grad_norm(E.parameters(), 0.25)
		torch.nn.utils.clip_grad_norm(D.parameters(), 0.25)
	
		e_optim.step()
#		a_optim.step()
		d_optim.step()
	print('Epoch {} Loss: {}'.format(epoch, total_loss/len(DS)))
	if epoch%SAVE == 0:
		torch.save({'encoder': E.state_dict(),'decoder': D.state_dict(), 'loss': total_loss/len(DS)}, os.path.join(directory, '{}.tar'.format(epoch)))
