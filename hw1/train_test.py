import editdistance
from utils import trim_and_map,normalize, test_trim
from model import Feature_Dataset,LSTM,C_RNN #bilstm and c_rnn
from config import SAVE_DIR,USE_CUDA 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import os
import numpy as np

def train(feature,label, epochs, model, layer, hidden, save,postfix, index2char, index2phone, phone_map, phone2index):
	dataset = Feature_Dataset(feature,'train')
	train_size = int(0.9*len(dataset))
	if feature == 'mfcc':
		feature_dim = 39
	elif feature == 'fbank':
		feature_dim = 69
	elif feature == 'all':
		feature_dim = 108

	print("Building model and optimizer...")
	if model == 'LSTM':
		train_model = LSTM(feature_dim,hidden,layer)
	elif model == 'C_RNN':
		group_size = 5 
		train_model = C_RNN(group_size,feature_dim,hidden,layer)
	elif model == 'BiLSTM':
		train_model = LSTM(feature_dim, hidden, layer, bi = True)
	
	if USE_CUDA:
		train_model = train_model.cuda()
	optimizer = optim.Adam(train_model.parameters(), lr = 0.005)
	#optimizer = optim.SGD(train_model.parameters(),lr = 0.1)
	criterion = nn.NLLLoss()
	if USE_CUDA:
		criterion = criterion.cuda() 

	for epoch in range(1,epochs+1):
		print("Epoch {}".format(epoch))
		epoch_loss = 0
		epoch_edit = 0
		for i in tqdm(range(1,train_size+1)):
			data = dataset[i-1]
			speaker = data[0]
		
			train_model.zero_grad()
			input_hidden = train_model.init_hidden()
			
			train_feature = Variable(data[1].float())
			output =  train_model(train_feature,input_hidden)
			
			output_seq = test_trim(index2char, index2phone, phone_map, phone2index, torch.max(output,1)[1].data.cpu().numpy())
			target_seq = trim_and_map(index2char,index2phone, phone_map, phone2index, [[int(l)] for l in label[speaker]])
			
			target = Variable(torch.from_numpy(np.array(label[speaker]).astype('int')))
			target = target.cuda() if USE_CUDA else target
			
			loss = criterion(output,target)
			edit = editdistance.eval(output_seq,target_seq)

			epoch_loss += loss.data[0]/train_size
			epoch_edit += edit/train_size
		
			loss.backward()
			optimizer.step()

		print("Negative log-likelihood: {}".format(epoch_loss))
		print("Edit distance: {} ".format(epoch_edit))
		val_loss = 0
		val_edit = 0
		for i in tqdm(range(train_size+1,len(dataset)+1)):
			data = dataset[i-1]
			speaker = data[0]
			val_feature = Variable(data[1].float())
			
			output = train_model(val_feature,train_model.init_hidden())
			target = Variable(torch.from_numpy(np.array(label[speaker]).astype('int')))
			target = target.cuda() if USE_CUDA else target
			
			val_loss += criterion(output,target).data[0]		
			output_seq = test_trim(index2char,index2phone, phone_map, phone2index,torch.max(output,1)[1].data.cpu().numpy())
			target_seq = trim_and_map(index2char,index2phone, phone_map, phone2index,[[int(l)] for l in label[speaker]])
				
			val_edit += editdistance.eval(output_seq,target_seq)
		print("Validation loss: {}".format(val_loss/(len(dataset)-train_size)))
		print("Validation edit distance: {}".format(val_edit/(len(dataset)-train_size)))

		if epoch%save == 0:
			directory = os.path.join(SAVE_DIR, feature, model, '{}-{}{}'.format(layer,hidden,postfix))
			if not os.path.exists(directory):
				os.makedirs(directory)
			torch.save({
				'model': train_model.state_dict(),
                		'opt': optimizer.state_dict(),
                		'val_loss': val_loss/(len(dataset)-train_size),
				'val_edit': val_edit/(len(dataset)-train_size),
				}, os.path.join(directory, '{}.tar'.format(epoch)))
	print("Finish training")

def test(test, feature, model, hidden, layer,  output, index2char, index2phone, phone_map, phone2index):
	ans = open(output,'w')
	ans.write('id,phone_sequence\n')
	test_set = Feature_Dataset(feature,'test')
	if feature == 'mfcc':
		feature_dim = 39
	elif feature == 'fbank':
		feature_dim = 69
	elif feature == 'all':
		feature_dim = 108
	
	if model == 'LSTM':
		test_model = LSTM(feature_dim, hidden, layer)
	elif model == 'BiLSTM':
		test_model = LSTM(feature_dim,hidden,layer,bi = True)
	elif model == 'C_RNN':
		group_size = 5
		test_model = C_RNN(group_size, feature_dim, hidden, layer)    
	
	checkpoint = torch.load(test)
	test_model.load_state_dict(checkpoint['model'])
	test_model.eval()
	if USE_CUDA:
		test_model = test_model.cuda()		
	for i in tqdm(range(1,len(test_set)+1)):
		data = test_set[i-1]
		speaker = data[0]
		test_feature = Variable(data[1].float())
		test_hidden = test_model.init_hidden()
		output = torch.max(test_model(test_feature,test_hidden),1)[1]
		result = test_trim(index2char,index2phone, phone_map, phone2index, output.data.cpu().numpy())
		ans.write('{},{}\n'.format(speaker,result))
	ans.close()

def train_loss(train_dir):
    loss_dict = {}
    edit_dict = {}
    for f in os.listdir(train_dir):
        checkpoint = torch.load(os.path.join(train_dir,f))
        loss_dict[f] = checkpoint['val_loss']
        edit_dict[f] = checkpoint['val_edit']
    print(loss_dict)
    print(edit_dict)
	
