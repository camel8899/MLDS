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
import sys
VOCAB_SIZE = 3004 
DATA_DIR = './MLDS_hw2_data/training_data/feat'
TEST_DIR = './MLDS_hw2_data/testing_data/feat'
SAVE_DIR = './save'
LABEL_PATH = './MLDS_hw2_data/training_label.json'
ID_PATH = './MLDS_hw2_data/testing_id.txt'
USE_CUDA = torch.cuda.is_available()
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
hidden_size = 256 
postfix = 'clip25_drop3_lr-3_3000'

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

if USE_CUDA:
	print("USE CUDA")
else:
	print("Suggest using cuda")

BATCH_SIZE = 1
V = Vocab(LABEL_PATH)
DS = Testset(TEST_DIR,ID_PATH)
#DS = TA_Dataset(DATA_DIR,LABEL_PATH)
CHECK_PATH = './save/S2VT_EnDe/256-clip25_drop3_lr-3_3000_softmax/{}.tar'.format(sys.argv[1])
checkpoint = torch.load(CHECK_PATH)
E = Encoder(4096,hidden_size)
E.load_state_dict(checkpoint['encoder'])
D = Decoder(2*hidden_size,hidden_size)
D.load_state_dict(checkpoint['decoder'])
E = E.cuda()
D = D.cuda()
E.eval()
D.eval()
MAX_LENGTH = 20

def output_sen(index_list,V):
	output = ""
	for i in index_list:
		if i == EOS_TOKEN:
			#output += " ."
			break
		elif i == BOS_TOKEN or i == PAD_TOKEN:
			continue
		else:
			output += V.index2word[i]+" "
	return output	

with open('output_{}.csv'.format(sys.argv[1]),'w') as f:
	for data in DS:
		caption_id = data[1]
		data = data[0]
		feat = Variable(torch.from_numpy(data).view(1,-1,4096).float()).cuda()
		encoder_hidden = E.init_hidden(1)
		decoder_hidden = D.init_hidden(1)
		decoder_padding = Variable(torch.zeros(1,feat.size()[1],hidden_size).cuda())
		encoder_output1, encoder_hidden = E(feat,encoder_hidden)
		decoder_result1, decoder_output1, decoder_hidden = D(torch.cat((decoder_padding,encoder_output1),2),decoder_hidden)

		embed = torch.zeros(1,1,hidden_size).cuda()
		embed = Variable(embed)

		encoder_padding = Variable(torch.zeros(1,1,4096).cuda())
		result = []
		for i in range(MAX_LENGTH):
			encoder_output2, encoder_hidden = E(encoder_padding, encoder_hidden)
			decoder_result2, embed, decoder_hidden = D(torch.cat((embed,encoder_output2),2),decoder_hidden)
			result.append(torch.max(decoder_result2,1)[1].data[0])
		print(output_sen(result,V))
		f.write('{},{}\n'.format(caption_id,output_sen(result,V)))
