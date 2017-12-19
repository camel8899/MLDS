import argparse
from utils import *
from train_test import train,test,train_loss
from config import USE_CUDA

def parse():
	parser = argparse.ArgumentParser(description='Sequence labeling model')
	parser.add_argument('-f', '--feature',default = 'fbank', choices = ['fbank','mfcc','all'],\
				help='Train the model with different feature, default is fbank.')
	parser.add_argument('-m','--model',default = 'LSTM', choices = ['LSTM','BiLSTM', 'C_RNN'],\
				help='Model type, default is LSTM.')
	parser.add_argument('-te', '--test', help='Test the saved model (enter the model path)')
	parser.add_argument('-l', '--loss', help='save dir')
	parser.add_argument('-o', '--output', help='Output file name')
	parser.add_argument('-e', '--epochs', type=int, default=100, help='Train the model with specified epochs')
	parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
	parser.add_argument('-hi', '--hidden', type=int, default=128, help='Hidden size in RNN')
	parser.add_argument('-s', '--save', type=float, default=10, help='Save every s epochs')
	parser.add_argument('-p', '--postfix', type=str, default='', help='Model postfix if you want')
	

	args = parser.parse_args()
	return args

def run(args):
	if USE_CUDA:
		print("Using cuda...")
	else:
		print("Suggest using cuda, break now...")

	phone_map = make_phone_map()
	phone2index,index2phone, index2char = make_phone_char()
	label = make_label(phone2index)
	if args.test:
		test(args.test, args.feature, args.model, args.hidden, args.layer, args.output, index2char, index2phone, phone_map, phone2index)
	elif args.loss:
		train_loss(args.loss)	
	else:
		train(args.feature, label,  args.epochs, args.model, args.layer, args.hidden, args.save, args.postfix, index2char, index2phone, phone_map, phone2index)



if __name__ == '__main__':
    args = parse()
    run(args)


