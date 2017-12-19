from config import DATA_DIR,USE_CUDA,PHONE_NUM
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import numpy as np

class Feature_Dataset(Dataset):
	def __init__(self,dataset,mode):
		print("Preparing dataset")
		self.dataset = dataset
		self.mode = mode
		if dataset != 'all':
			filename = os.path.join(DATA_DIR,dataset,mode+'.ark')
		else:
			filename = os.path.join(DATA_DIR,'mfcc',mode+'.ark')
			fbank_file = os.path.join(DATA_DIR,'fbank',mode+'.ark')

		self.feature = defaultdict(list)
		with open(filename,'r') as f:
			for line in f.readlines():
				info = line.strip().split()
				speaker = '_'.join(info[0].split('_')[:-1])
				self.feature[speaker].append(info[1:])	
		if dataset == 'all':
			with open(fbank_file,'r') as f:
				for line in f.readlines():
					info = line.strip().split()
					speaker = '_'.join(info[0].split('_')[:-1])
					self.feature[speaker].append(info[1:])
			for speaker,feature in self.feature.items():
				seq_len = int(len(self.feature[speaker])/2)
				for i in range(seq_len):
					self.feature[speaker][i] += self.feature[speaker][i+seq_len]
				del self.feature[speaker][seq_len:]
	def __len__(self):
		return len(self.feature)
	
	def __getitem__(self,index):
		chosen_speaker = list(self.feature.keys())[index]
		feature = torch.from_numpy(np.array(self.feature[chosen_speaker]).astype('float'))
		feature = feature.cuda() if USE_CUDA else feature
		return [chosen_speaker,feature]
    
class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, layers=1,bi=False):
		super(LSTM, self).__init__()
		self.layers = layers	
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, bidirectional=bi, dropout = 0.4)
		self.hidden2out = nn.Linear(hidden_size,PHONE_NUM)
		self.dropout = nn.Dropout(0.4)
		self.softmax = nn.LogSoftmax()
		self.bi = bi	
	def init_hidden(self):
		direction = 2 if self.bi else 1
		if USE_CUDA:
			return (Variable(torch.zeros(self.layers*direction, 1, self.hidden_size)).cuda(),\
				Variable(torch.zeros(self.layers*direction, 1, self.hidden_size)).cuda())
		else:
			return (Variable(torch.zeros(self.layers*direction, 1, self.hidden_size)),\
				Variable(torch.zeros(self.layers*direction, 1, self.hidden_size))) 
 
	def forward(self, input_seq, hidden):
		input_seq = input_seq.view(len(input_seq),1,-1)
		output, hidden = self.lstm(input_seq,hidden) 
		if self.bi:
			output = (output[:,:,:self.hidden_size]+output[:,:,self.hidden_size:])/2
		output = self.dropout(output)
		output = self.hidden2out(output.view(output.size()[0],-1))
		output = self.softmax(output)
		return output
    
class C_RNN(nn.Module):
	def __init__(self, group_size, input_size, hidden_size, layers=1,bi=False):
		super(C_RNN, self).__init__()
		self.group_size = group_size
		self.feature_len = 5 
		self.filter = 10 
		self.layers = layers	
		self.hidden_size = hidden_size
		self.cnn = nn.Conv2d(1,self.filter,kernel_size = (self.group_size,self.feature_len))
		self.pooling = nn.MaxPool2d((1,3))
		self.lstm = nn.LSTM(249, hidden_size, num_layers=layers, bidirectional=bi, dropout = 0.5)
		self.hidden2out = nn.Linear(hidden_size,PHONE_NUM)
		self.softmax = nn.LogSoftmax()
	
	def init_hidden(self):
		if USE_CUDA:
			return (Variable(torch.rand(self.layers, 1, self.hidden_size)).cuda(),\
				Variable(torch.rand(self.layers, 1, self.hidden_size)).cuda())
		else:
			return (Variable(torch.rand(self.layers, 1, self.hidden_size)),\
				Variable(torch.rand(self.layers, 1, self.hidden_size))) 
 
	def forward(self, input_seq, hidden):
		padding_size = int(self.group_size/2)
		input_seq = torch.cat((input_seq[0].repeat(padding_size,1),input_seq),0)
		input_seq = torch.cat((input_seq,input_seq[-1].repeat(padding_size,1)),0)
		for i in range(len(input_seq)-self.group_size+1):
			feature = input_seq[i:i+self.group_size,39:]
			feature = feature.contiguous().view(1,1,feature.size()[0],feature.size()[1])
			if i == 0:
				input_feature = self.cnn(feature)
				input_feature = self.pooling(input_feature)
				input_feature = input_feature.view(1,self.filter*input_feature.size()[-1])
			else:
				new_feature = self.cnn(feature)
				new_feature = self.pooling(new_feature)
				new_feature = new_feature.view(1,self.filter*new_feature.size()[-1])
				input_feature = torch.cat((input_feature,new_feature),0)
		
		input_feature = input_feature.view(input_feature.size()[0],-1)
		input_feature = torch.cat((input_feature,input_seq[padding_size:len(input_seq)-padding_size,:39]),1)
		input_feature = input_feature.view(input_feature.size()[0],1,-1)
		output, hidden = self.lstm(input_feature,hidden) 
		output = self.hidden2out(output.view(output.size()[0],-1))
		output = self.softmax(output)
		return output

