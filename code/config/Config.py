# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F





IGNORE_INDEX = -100
is_transformer = False



#这里是我的了
import argparse
import datetime

import torch
import ujson as json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from .args import add_args
from .model import DocREModel
from .utils import set_seed, collate_fn, create_directory
from .prepro import read_docred
from .evaluation import to_official, official_evaluate, merge_results

from tqdm import tqdm

import pandas as pd
import pickle

#因为导入模块会出现问题，所以我决定试一试把上面导入的函数全都直接复制进来

def load_input(batch, device, tag="dev"):

    input = {'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device),
            'labels': batch[2].to(device),
            'entity_pos': batch[3],
            'hts': batch[4],
            'sent_pos': batch[5],
            'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
            'teacher_attns': batch[7].to(device) if not batch[7] is None else None,
            'tag': tag
            } 

    return input

def train(args, model, train_features, dev_features):

    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler()
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator, desc='Train epoch'):
            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                optimizer.zero_grad()
                model.train()

                inputs = load_input(batch, args.device)  
                outputs = model(**inputs)
                loss = 0

                if inputs["sent_labels"] != None:
                    loss += outputs["loss"]["evi_loss"] * args.evi_lambda
                
                loss = sum(loss) / args.gradient_accumulation_steps
                
                return loss

                     

        

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    
    final_loss = 0 #到这里，都是我修改的东西了，希望成功
    
    final_loss = finetune(train_features, optimizer, args.num_train_epochs, num_steps)
    
    return final_loss

def evaluate(args, model, features, tag="dev"):
    
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, evi_preds = [], []
    scores, topks = [], []
    attns = []
    
    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()

        if args.save_attn:
            tag = "infer"

        inputs = load_input(batch, args.device, tag)

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs["rel_pred"]
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

            if "scores" in outputs:
                scores.append(outputs["scores"].cpu().numpy())  
                topks.append(outputs["topks"].cpu().numpy())   

            if "evi_pred" in outputs: # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]
                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)   
            
            if "attns" in outputs: # attention recorded
                attn = outputs["attns"]
                attns.extend([a.cpu().numpy() for a in attn])


    preds = np.concatenate(preds, axis=0)

    if scores != []:
        scores = np.concatenate(scores, axis=0)
        topks =  np.concatenate(topks, axis=0)

    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)
    
    official_results, results = to_official(preds, features, evi_preds = evi_preds, scores = scores, topks = topks)
    
    if len(official_results) > 0:
        if tag == "dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file, args.dev_file)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file, args.test_file)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign], 
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {"dev_F1": best_re[-1] * 100, "dev_evi_F1": best_evi[-1] * 100, "dev_F1_ign": best_re_ign[-1] * 100}

    if args.save_attn:
        
        attns_path = os.path.join(args.load_path, f"{os.path.splitext(args.test_file)[0]}.attns")        
        print(f"saving attentions into {attns_path} ...")
        with open(attns_path, "wb") as f:
            pickle.dump(attns, f)

    return scores, output, official_results, results

def dump_to_file(offi:list, offi_path: str, scores: list, score_path: str, results: list = [], res_path: str = "", thresh: float = None):
    '''
    dump scores and (top-k) predictions to file.
    
    '''
    print(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))
    
    print(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns = headers)
    print(scores_pd)
    scores_pd.to_csv(score_path, sep='\t')

    if len(results) != 0:
        assert res_path != ""
        print(f"saving topk results into {res_path} ...")
        json.dump(results, open(res_path, "w"))
    
    if thresh != None:
        thresh_path = os.path.join(os.path.dirname(offi_path), "thresh")
        if not os.path.exists(thresh_path):
            print(f"saving threshold into {thresh_path} ...")
            json.dump(thresh, open(thresh_path, "w"))        

    return


def calculate():
    
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
        

    

    # create directory to save checkpoints and predicted files
    time = str(datetime.datetime.now()).replace(' ','_')
    save_path_ = os.path.join(args.save_path, f"{time}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    model0 = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.transformer_type = args.transformer_type

    set_seed(args)
    
    read = read_docred    
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model0 = DocREModel(config, model0, tokenizer,
                    num_labels=args.num_labels,
                    max_sent_num=args.max_sent_num, 
                    evi_thresh=args.evi_thresh)

    model0.to(args.device)

    # load model from existing checkpoint记得添加模型路径
    model0.eval()
    
    
    

    model_path = os.path.join(args.load_path, "best.ckpt")
    model0.load_state_dict(torch.load(model_path))

    my_loss = 0
        
    create_directory(save_path_)
    args.save_path = save_path_
    
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)

    train_features = read(train_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
    dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length)

    my_loss = train(args, model0, train_features, dev_features)

    return my_loss










class Accuracy(object):
	def __init__(self):
		self.correct = 0
		self.total = 0
	def add(self, is_correct):
		self.total += 1
		if is_correct:
			self.correct += 1
	def get(self):
		if self.total == 0:
			return 0.0
		else:
			return float(self.correct) / self.total
	def clear(self):
		self.correct = 0
		self.total = 0 

class Config(object):
	def __init__(self, args):
		self.acc_NA = Accuracy()
		self.acc_not_NA = Accuracy()
		self.acc_total = Accuracy()
		self.data_path = './prepro_data'
		self.use_bag = False
		self.use_gpu = True
		self.is_training = True
		self.max_length = 512
		self.pos_num = 2 * self.max_length
		self.entity_num = self.max_length
		self.relation_num = 97

		self.coref_size = 20
		self.entity_type_size = 20
		self.max_epoch = 20
		self.opt_method = 'Adam'
		self.optimizer = None

		self.checkpoint_dir = './checkpoint'
		self.fig_result_dir = './fig_result'
		self.test_epoch = 5
		self.pretrain_model = None


		self.word_size = 100
		self.epoch_range = None
		self.cnn_drop_prob = 0.5  # for cnn
		self.keep_prob = 0.8  # for lstm

		self.period = 50

		self.batch_size = 40
		self.h_t_limit = 1800

		self.test_batch_size = self.batch_size
		self.test_relation_limit = 1800
		self.char_limit = 16
		self.sent_limit = 25
		self.dis2idx = np.zeros((512), dtype='int64')
		self.dis2idx[1] = 1
		self.dis2idx[2:] = 2
		self.dis2idx[4:] = 3
		self.dis2idx[8:] = 4
		self.dis2idx[16:] = 5
		self.dis2idx[32:] = 6
		self.dis2idx[64:] = 7
		self.dis2idx[128:] = 8
		self.dis2idx[256:] = 9
		self.dis_size = 20

		self.train_prefix = 'dev_train'
		self.test_prefix = 'dev_dev'


		if not os.path.exists("log"):
			os.mkdir("log")

	def set_data_path(self, data_path):
		self.data_path = data_path
	def set_max_length(self, max_length):
		self.max_length = max_length
		self.pos_num = 2 * self.max_length
	def set_num_classes(self, num_classes):
		self.num_classes = num_classes
	def set_window_size(self, window_size):
		self.window_size = window_size
	def set_word_size(self, word_size):
		self.word_size = word_size
	def set_max_epoch(self, max_epoch):
		self.max_epoch = max_epoch
	def set_batch_size(self, batch_size):
		self.batch_size = batch_size
	def set_opt_method(self, opt_method):
		self.opt_method = opt_method
	def set_drop_prob(self, drop_prob):
		self.drop_prob = drop_prob
	def set_checkpoint_dir(self, checkpoint_dir):
		self.checkpoint_dir = checkpoint_dir
	def set_test_epoch(self, test_epoch):
		self.test_epoch = test_epoch
	def set_pretrain_model(self, pretrain_model):
		self.pretrain_model = pretrain_model
	def set_is_training(self, is_training):
		self.is_training = is_training
	def set_use_bag(self, use_bag):
		self.use_bag = use_bag
	def set_use_gpu(self, use_gpu):
		self.use_gpu = use_gpu
	def set_epoch_range(self, epoch_range):
		self.epoch_range = epoch_range
	
	def load_train_data(self):
		print("Reading training data...")
		prefix = self.train_prefix

		print ('train', prefix)
		self.data_train_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
		self.data_train_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
		self.data_train_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
		self.data_train_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
		self.train_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))

		print("Finish reading")

		self.train_len = ins_num = self.data_train_word.shape[0]
		assert(self.train_len==len(self.train_file))

		self.train_order = list(range(ins_num))
		self.train_batches = ins_num // self.batch_size
		if ins_num % self.batch_size != 0:
			self.train_batches += 1

	def load_test_data(self):
		print("Reading testing data...")
		self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
		self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
		self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
		self.id2rel = {v: k for k,v in self.rel2id.items()}

		prefix = self.test_prefix
		print (prefix)
		self.is_test = ('dev_test' == prefix)
		self.data_test_word = np.load(os.path.join(self.data_path, prefix+'_word.npy'))
		self.data_test_pos = np.load(os.path.join(self.data_path, prefix+'_pos.npy'))
		self.data_test_ner = np.load(os.path.join(self.data_path, prefix+'_ner.npy'))
		self.data_test_char = np.load(os.path.join(self.data_path, prefix+'_char.npy'))
		self.test_file = json.load(open(os.path.join(self.data_path, prefix+'.json')))


		self.test_len = self.data_test_word.shape[0]
		assert(self.test_len==len(self.test_file))


		print("Finish reading")

		self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
		if self.data_test_word.shape[0] % self.test_batch_size != 0:
			self.test_batches += 1

		self.test_order = list(range(self.test_len))
		self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)


	def get_train_batch(self):
		random.shuffle(self.train_order)

		context_idxs = torch.LongTensor(self.batch_size, self.max_length).cuda()
		context_pos = torch.LongTensor(self.batch_size, self.max_length).cuda()
		h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
		t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).cuda()
		relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cuda()
		relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cuda()

		pos_idx = torch.LongTensor(self.batch_size, self.max_length).cuda()

		context_ner = torch.LongTensor(self.batch_size, self.max_length).cuda()
		context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).cuda()

		relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()


		ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).cuda()

		for b in range(self.train_batches):
			start_id = b * self.batch_size
			cur_bsz = min(self.batch_size, self.train_len - start_id)
			cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
			cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x]>0) , reverse = True)

			for mapping in [h_mapping, t_mapping]:
				mapping.zero_()

			for mapping in [relation_multi_label, relation_mask, pos_idx]:
				mapping.zero_()

			ht_pair_pos.zero_()


			relation_label.fill_(IGNORE_INDEX)

			max_h_t_cnt = 1


			for i, index in enumerate(cur_batch):
				context_idxs[i].copy_(torch.from_numpy(self.data_train_word[index, :]))
				context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))
				context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))

				for j in range(self.max_length):
					if self.data_train_word[index, j]==0:
						break
					pos_idx[i, j] = j+1

				ins = self.train_file[index]
				labels = ins['labels']
				idx2label = defaultdict(list)

				for label in labels:
					idx2label[(label['h'], label['t'])].append(label['r'])



				train_tripe = list(idx2label.keys())
				for j, (h_idx, t_idx) in enumerate(train_tripe):
					hlist = ins['vertexSet'][h_idx]
					tlist = ins['vertexSet'][t_idx]

					for h in hlist:
						h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

					for t in tlist:
						t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

					label = idx2label[(h_idx, t_idx)]

					delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
					if delta_dis < 0:
						ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
					else:
						ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])


					for r in label:
						relation_multi_label[i, j, r] = 1

					relation_mask[i, j] = 1
					rt = np.random.randint(len(label))
					relation_label[i, j] = label[rt]



				lower_bound = len(ins['na_triple'])
				# random.shuffle(ins['na_triple'])
				# lower_bound = max(20, len(train_tripe)*3)


				for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_tripe)):
					hlist = ins['vertexSet'][h_idx]
					tlist = ins['vertexSet'][t_idx]

					for h in hlist:
						h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])

					for t in tlist:
						t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

					relation_multi_label[i, j, 0] = 1
					relation_label[i, j] = 0
					relation_mask[i, j] = 1
					delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
					if delta_dis < 0:
						ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
					else:
						ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

				max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)


			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())

			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
				   'input_lengths' : input_lengths,
				   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
				   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt],
				   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
				   }

	def get_test_batch(self):
		context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
		context_pos = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
		h_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
		t_mapping = torch.Tensor(self.test_batch_size, self.test_relation_limit, self.max_length).cuda()
		context_ner = torch.LongTensor(self.test_batch_size, self.max_length).cuda()
		context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).cuda()
		relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).cuda()
		ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).cuda()

		for b in range(self.test_batches):
			start_id = b * self.test_batch_size
			cur_bsz = min(self.test_batch_size, self.test_len - start_id)
			cur_batch = list(self.test_order[start_id : start_id + cur_bsz])

			for mapping in [h_mapping, t_mapping, relation_mask]:
				mapping.zero_()


			ht_pair_pos.zero_()

			max_h_t_cnt = 1

			cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x]>0) , reverse = True)

			labels = []

			L_vertex = []
			titles = []
			indexes = []
			for i, index in enumerate(cur_batch):
				context_idxs[i].copy_(torch.from_numpy(self.data_test_word[index, :]))
				context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
				context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
				context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))



				idx2label = defaultdict(list)
				ins = self.test_file[index]

				for label in ins['labels']:
					idx2label[(label['h'], label['t'])].append(label['r'])



				L = len(ins['vertexSet'])
				titles.append(ins['title'])

				j = 0
				for h_idx in range(L):
					for t_idx in range(L):
						if h_idx != t_idx:
							hlist = ins['vertexSet'][h_idx]
							tlist = ins['vertexSet'][t_idx]

							for h in hlist:
								h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
							for t in tlist:
								t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

							relation_mask[i, j] = 1

							delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
							if delta_dis < 0:
								ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
							else:
								ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
							j += 1


				max_h_t_cnt = max(max_h_t_cnt, j)
				label_set = {}
				for label in ins['labels']:
					label_set[(label['h'], label['t'], label['r'])] = label['in'+self.train_prefix]

				labels.append(label_set)


				L_vertex.append(L)
				indexes.append(index)



			input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
			max_c_len = int(input_lengths.max())


			yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
				   'h_mapping': h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   't_mapping': t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
				   'labels': labels,
				   'L_vertex': L_vertex,
				   'input_lengths': input_lengths,
				   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
				   'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
				   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
				   'titles': titles,
				   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt],
				   'indexes': indexes
				   }

	def train(self, model_pattern, model_name):

		ori_model = model_pattern(config = self)
		if self.pretrain_model != None:
			ori_model.load_state_dict(torch.load(self.pretrain_model))
		ori_model.cuda()
		model = nn.DataParallel(ori_model)

		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
		# nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
		BCE = nn.BCEWithLogitsLoss(reduction='none')

		if not os.path.exists(self.checkpoint_dir):
			os.mkdir(self.checkpoint_dir)

		best_auc = 0.0
		best_f1 = 0.0
		best_epoch = 0

		model.train()

		global_step = 0
		total_loss = 0
		start_time = time.time()

		def logging(s, print_=True, log_=True):
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')

		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.ylim(0.3, 1.0)
		plt.xlim(0.0, 0.4)
		plt.title('Precision-Recall')
		plt.grid(True)

		for epoch in range(self.max_epoch):

			self.acc_NA.clear()
			self.acc_not_NA.clear()
			self.acc_total.clear()

			for data in self.get_train_batch():

				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				h_mapping = data['h_mapping']
				t_mapping = data['t_mapping']
				relation_label = data['relation_label']
				input_lengths =  data['input_lengths']
				relation_multi_label = data['relation_multi_label']
				relation_mask = data['relation_mask']
				context_ner = data['context_ner']
				context_char_idxs = data['context_char_idxs']
				ht_pair_pos = data['ht_pair_pos']
                
				#sent_h_mapping = data['sent_h_mapping']
				#print(sent_h_mapping,"你好！")

				dis_h_2_t = ht_pair_pos+10
				dis_t_2_h = -ht_pair_pos+10


				predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)
                
				#print(predict_re)
				#print(relation_multi_label)
				#print("------")
                
				loss = torch.sum(BCE(predict_re, relation_multi_label)*relation_mask.unsqueeze(2)) /  (self.relation_num * torch.sum(relation_mask))
                

                
				print(loss)
                
				output = torch.argmax(predict_re, dim=-1)
				output = output.data.cpu().numpy()

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				relation_label = relation_label.data.cpu().numpy()

				for i in range(output.shape[0]):
					for j in range(output.shape[1]):
						label = relation_label[i][j]
						if label<0:
							break

						if label == 0:
							self.acc_NA.add(output[i][j] == label)
						else:
							self.acc_not_NA.add(output[i][j] == label)

						self.acc_total.add(output[i][j] == label)

				global_step += 1
				total_loss += loss.item()

				if global_step % self.period == 0 :
					cur_loss = total_loss / self.period
					elapsed = time.time() - start_time
					logging('| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()))
					total_loss = 0
					start_time = time.time()



			if (epoch+1) % self.test_epoch == 0:
				logging('-' * 89)
				eval_start_time = time.time()
				model.eval()
				f1, auc, pr_x, pr_y = self.test(model, model_name)
				model.train()
				logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
				logging('-' * 89)


				if f1 > best_f1:
					best_f1 = f1
					best_auc = auc
					best_epoch = epoch
					path = os.path.join(self.checkpoint_dir, model_name)
					torch.save(ori_model.state_dict(), path)

					plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
					plt.legend(loc="upper right")
					plt.savefig(os.path.join("fig_result", model_name))

		print("Finish training")
		print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
		print("Storing best result...")
		print("Finish storing")

	def test(self, model, model_name, output=False, input_theta=-1):
		data_idx = 0
		eval_start_time = time.time()
		# test_result_ignore = []
		total_recall_ignore = 0

		test_result = []
		total_recall = 0
		top1_acc = have_label = 0

		def logging(s, print_=True, log_=True):
			if print_:
				print(s)
			if log_:
				with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
					f_log.write(s + '\n')



		for data in self.get_test_batch():
			with torch.no_grad():
				context_idxs = data['context_idxs']
				context_pos = data['context_pos']
				h_mapping = data['h_mapping']
				t_mapping = data['t_mapping']
				labels = data['labels']
				L_vertex = data['L_vertex']
				input_lengths =  data['input_lengths']
				context_ner = data['context_ner']
				context_char_idxs = data['context_char_idxs']
				relation_mask = data['relation_mask']
				ht_pair_pos = data['ht_pair_pos']

				titles = data['titles']
				indexes = data['indexes']

				dis_h_2_t = ht_pair_pos+10
				dis_t_2_h = -ht_pair_pos+10

				predict_re = model(context_idxs, context_pos, context_ner, context_char_idxs, input_lengths,
								   h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h)

				predict_re = torch.sigmoid(predict_re)

			predict_re = predict_re.data.cpu().numpy()

			for i in range(len(labels)):
				label = labels[i]
				index = indexes[i]


				total_recall += len(label)
				for l in label.values():
					if not l:
						total_recall_ignore += 1

				L = L_vertex[i]
				j = 0

				for h_idx in range(L):
					for t_idx in range(L):
						if h_idx != t_idx:
							r = np.argmax(predict_re[i, j])
							if (h_idx, t_idx, r) in label:
								top1_acc += 1

							flag = False

							for r in range(1, self.relation_num):
								intrain = False

								if (h_idx, t_idx, r) in label:
									flag = True
									if label[(h_idx, t_idx, r)]==True:
										intrain = True


								# if not intrain:
								# 	test_result_ignore.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

								test_result.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]), intrain,  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

							if flag:
								have_label += 1

							j += 1


			data_idx += 1

			if data_idx % self.period == 0:
				print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
				eval_start_time = time.time()

		# test_result_ignore.sort(key=lambda x: x[1], reverse=True)
		test_result.sort(key = lambda x: x[1], reverse=True)

		print ('total_recall', total_recall)
		# plt.xlabel('Recall')
		# plt.ylabel('Precision')
		# plt.ylim(0.2, 1.0)
		# plt.xlim(0.0, 0.6)
		# plt.title('Precision-Recall')
		# plt.grid(True)

		pr_x = []
		pr_y = []
		correct = 0
		w = 0

		if total_recall == 0:
			total_recall = 1  # for test

		for i, item in enumerate(test_result):
			correct += item[0]
			pr_y.append(float(correct) / (i + 1))
			pr_x.append(float(correct) / total_recall)
			if item[1] > input_theta:
				w = i


		pr_x = np.asarray(pr_x, dtype='float32')
		pr_y = np.asarray(pr_y, dtype='float32')
		f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		f1 = f1_arr.max()
		f1_pos = f1_arr.argmax()
		theta = test_result[f1_pos][1]

		if input_theta==-1:
			w = f1_pos
			input_theta = theta

		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)
		if not self.is_test:
			logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
		else:
			logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

		if output:
			# output = [x[-4:] for x in test_result[:w+1]]
			output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6]} for x in test_result[:w+1]]
			json.dump(output, open(self.test_prefix + "_index.json", "w"))

		# plt.plot(pr_x, pr_y, lw=2, label=model_name)
		# plt.legend(loc="upper right")
		if not os.path.exists(self.fig_result_dir):
			os.mkdir(self.fig_result_dir)
		# plt.savefig(os.path.join(self.fig_result_dir, model_name))

		pr_x = []
		pr_y = []
		correct = correct_in_train = 0
		w = 0
		for i, item in enumerate(test_result):
			correct += item[0]
			if item[0] & item[2]:
				correct_in_train += 1
			if correct_in_train==correct:
				p = 0
			else:
				p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
			pr_y.append(p)
			pr_x.append(float(correct) / total_recall)
			if item[1] > input_theta:
				w = i

		pr_x = np.asarray(pr_x, dtype='float32')
		pr_y = np.asarray(pr_y, dtype='float32')
		f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
		f1 = f1_arr.max()

		auc = sklearn.metrics.auc(x = pr_x, y = pr_y)

		logging('Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta, f1_arr[w], auc))

		return f1, auc, pr_x, pr_y



	def testall(self, model_pattern, model_name, input_theta):#, ignore_input_theta):
		model = model_pattern(config = self)

		model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, model_name)))
		model.cuda()
		model.eval()
		f1, auc, pr_x, pr_y = self.test(model, model_name, True, input_theta)
