import sys
import os
sys.path.append('../code/')
import torch
import config as config

from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer,BertTokenizerFast
import json
def eval_link(test_loader,eval_fun):
    model = torch.load(config.save_train_model_file)
    model.eval()
    dev_losses = 0
    tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
    idx2label = ['nolink','qslink','olink','movelink']

    def init_res(save_res):
        save_res['seq'] = []
        save_res['e1'] = []
        save_res['tr'] = []
        save_res['e2'] = []
        save_res['gold_label'] = []
        save_res['pre_label'] = []

    save_res_lis = [] 
    with torch.no_grad():
        prey_lis=[]
        truy_lis=[]
        for idx, batch_samples in enumerate(test_loader):
            cbatch=batch_samples
            loss,prey = model(**cbatch)
            
            prey_lab=torch.argmax(prey,dim=-1)
            
            prey_lis += prey_lab.to('cpu').numpy().tolist()
            
            truy_lis += cbatch['label'].view(-1).to('cpu').numpy().tolist()

            for idx,seq in enumerate(cbatch['bert_data']):
                cdict = dict()
                init_res(cdict)
                cdict['seq'] = tokenizer.decode(seq,skip_special_tokens=True)
                cdict['e1'] = tokenizer.decode(seq[cbatch['tm_tok_sted'][idx][0]:cbatch['tm_tok_sted'][idx][1]+1])
                cdict['tr'] = tokenizer.decode(seq[cbatch['tr_tok_sted'][idx][0]:cbatch['tr_tok_sted'][idx][1]+1])
                cdict['e2'] = tokenizer.decode(seq[cbatch['lg_tok_sted'][idx][0]:cbatch['lg_tok_sted'][idx][1]+1])
                cdict['gold_label'] = idx2label[cbatch['label'][idx].item()]
                cdict['pre_label'] = idx2label[prey_lis[idx]]
                save_res_lis.append(cdict)
                
                
            
            dev_losses += loss.item()


    metrics = {}
    metrics['f1']=eval_fun(truy_lis,prey_lis)
    class_report= classification_report(truy_lis, prey_lis,
                                                target_names=['no-link','qslink','olink','movelink'])
    id2label = ['no-link','qslink','olink','movelink']
    truy_lis = [id2label[i] for i in truy_lis]
    prey_lis = [id2label[i] for i in prey_lis]
    confusion_mat = confusion_matrix(truy_lis, prey_lis,labels = ['no-link','qslink','olink','movelink'])
    print(class_report)
    print(confusion_mat)
    metrics['loss'] = float(dev_losses) / len(test_loader)

    print('testf1: ',metrics['f1'])
    print('loss: ',metrics['loss'])
    '''
        保存矩阵
    '''
    if(os.path.exists(config.save_train_result_dir)):
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n'+str(class_report))
    else:
        os.makedirs(config.save_train_result_dir)
        with open (config.save_train_result_file,'w')as f:
            f.write(str(metrics)+'\n'+str(class_report))
    '''
        保存预测结果
    '''
    if(os.path.exists(config.save_test_all_res_dir)):
        with open (config.save_res_file,'w')as f:
            # json.dump(save_res,f)
            for seq_pre_data in save_res_lis:
                json_str=json.dumps(seq_pre_data,ensure_ascii=False)
                f.write(json_str+'\n')
    else:
        os.makedirs(config.save_test_all_res_dir)
        with open (config.save_res_file,'w')as f:
            # json.dump(save_res,f)
            for seq_pre_data in save_res_lis:
                json_str=json.dumps(seq_pre_data,ensure_ascii=False)
                f.write(json_str+'\n')
    test_pre_data=[]
    with open (config.save_res_file,'r')as f:
        for lin in f.readlines():
            test_pre_data.append(json.loads(lin.strip()))
    print(test_pre_data[0])

    
    