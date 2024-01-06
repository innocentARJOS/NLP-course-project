import argparse
import os
import datetime

import numpy as np
import torch
import ujson as json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from args import add_args
from model import DocREModel
from utils import set_seed, collate_fn, create_directory
from prepro import read_docred
from evaluation import to_official, official_evaluate, merge_results

from tqdm import tqdm

import pandas as pd
import pickle

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
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.transformer_type = args.transformer_type

    set_seed(args)
    
    read = read_docred    
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = DocREModel(config, model, tokenizer,
                    num_labels=args.num_labels,
                    max_sent_num=args.max_sent_num, 
                    evi_thresh=args.evi_thresh)
    model.to(args.device)

    # load model from existing checkpoint记得添加模型路径

    model_path = os.path.join(args.load_path, "best.ckpt")
    model.load_state_dict(torch.load(model_path))

    my_loss = 0
        
    create_directory(save_path_)
    args.save_path = save_path_
    
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)

    train_features = read(train_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
    dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type, max_seq_length=args.max_seq_length)

    my_loss = train(args, model, train_features, dev_features)

    return my_loss


if __name__ == "__main__":
    main()
