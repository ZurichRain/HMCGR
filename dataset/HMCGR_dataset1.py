import os
import sys
sys.path.append('../code/')
from torch.utils.data import Dataset
from transformers import BertTokenizer,BertTokenizerFast,T5Tokenizer
import torch
import config
from util_script.util import get_char2tok_spanlis_one_seq
import dgl

class HMCGRDataset(Dataset):
    def __init__(self, seq_data_lis):
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config.bert_pathname)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(config.T5base_pathname)
        self.seq_data_lis = seq_data_lis
        self.device = config.device

        self.dataset = self.preprocess()
    
    def preprocess(self):
        self.get_seq_bert_tok()
        self.get_seq_t5_tok()
        self.get_rfx_seq_tok()
        self.get_elements_bert_tok_sted()
        self.get_graph()

        '''
            a sentence as a graph
        '''

        input_data=[]
        for data in self.seq_data_lis:
            cur_dict=dict()
            cur_dict['seq_bert_tok'] = data['seq_bert_tok']
            cur_dict['tm_tok_sted'] = data['tm_tok_sted']
            cur_dict['tr_tok_sted'] = data['tr_tok_sted']
            cur_dict['lg_tok_sted'] = data['lg_tok_sted']
            cur_dict['tm_nodelis'] = data['tm_nodelis']
            cur_dict['tr_nodelis'] = data['tr_nodelis']
            cur_dict['lg_nodelis'] = data['lg_nodelis']
            cur_dict['node_to_tokid'] = data['node_to_tokid']
            cur_dict['graph'] = data['graph']
            cur_dict['seq_t5_tok'] = data['seq_t5_tok']
            cur_dict['target_seq_t5_tok'] = data['target_seq_t5_tok']
            cur_dict['seq_rfx_tok'] = data['seq_rfx_tok']

            cur_dict['label'] = data['link'][-1]

            input_data.append(cur_dict)


        return input_data
        

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def get_seq_bert_tok(self):
        for seq_data in self.seq_data_lis:
            seq_data['seq_bert_tok']=self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(seq_data['text']))
    def get_seq_t5_tok(self):
        for seq_data in self.seq_data_lis:
            seq_data['seq_t5_tok']=self.t5_tokenizer.convert_tokens_to_ids(self.t5_tokenizer.tokenize(seq_data['text']))
            if seq_data['link'][-1]==1:
                target_seq = 'qualitative link can be describe as following: trajector is {}, trigger is {}, landmark is {}'.format(\
                    seq_data['text'][seq_data['link'][0][0]:seq_data['link'][0][1]],\
                    seq_data['text'][seq_data['link'][1][0]:seq_data['link'][1][1]],\
                    seq_data['text'][seq_data['link'][2][0]:seq_data['link'][2][1]])
            elif seq_data['link'][-1]==2:
                target_seq = 'orientation link can be describe as following: trajector is {}, trigger is {}, landmark is {}'.format(\
                    seq_data['text'][seq_data['link'][0][0]:seq_data['link'][0][1]],\
                    seq_data['text'][seq_data['link'][1][0]:seq_data['link'][1][1]],\
                    seq_data['text'][seq_data['link'][2][0]:seq_data['link'][2][1]])
            else:
                target_seq = 'move link can be describe as following: mover is {}, trigger is {}, goal is {}'.format(\
                    seq_data['text'][seq_data['link'][0][0]:seq_data['link'][0][1]],\
                    seq_data['text'][seq_data['link'][1][0]:seq_data['link'][1][1]],\
                    seq_data['text'][seq_data['link'][2][0]:seq_data['link'][2][1]])
            seq_data['target_seq_t5_tok']=self.t5_tokenizer.convert_tokens_to_ids(self.t5_tokenizer.tokenize(target_seq))

    def get_rfx_tr(self,tr_text):
        c_tr_text = tr_text.lower()
        rfxtr = dict()
        rfxtr['up'] = 'down'
        rfxtr[c_tr_text] = c_tr_text
        return rfxtr[c_tr_text]
    def get_rfx_seq_tok(self):
        for seq_data in self.seq_data_lis:
            ori_text = seq_data['text']
            tm = ori_text[seq_data['link'][0][0]:seq_data['link'][0][1]]
            tr = ori_text[seq_data['link'][1][0]:seq_data['link'][1][1]]
            lg = ori_text[seq_data['link'][2][0]:seq_data['link'][2][1]]
            stsort = sorted([(seq_data['link'][0][0],seq_data['link'][0][1]),\
                            (seq_data['link'][1][0],seq_data['link'][1][1]),\
                            (seq_data['link'][2][0],seq_data['link'][2][1])])
            rfx_text = ori_text[:stsort[0][0]]
            if(stsort[0][0]==seq_data['link'][0][0]):
                rfx_text += lg
            elif(stsort[0][0]==seq_data['link'][2][0]):
                rfx_text += tm
            else:
                rfx_text += self.get_rfx_tr(tr)

            rfx_text += ori_text[stsort[0][1]:stsort[1][0]]
            if(stsort[1][0]==seq_data['link'][0][0]):
                rfx_text += lg
            elif(stsort[1][0]==seq_data['link'][2][0]):
                rfx_text += tm
            else:
                rfx_text += self.get_rfx_tr(tr)

            rfx_text += ori_text[stsort[1][1]:stsort[2][0]]
            if(stsort[2][0]==seq_data['link'][0][0]):
                rfx_text += lg
            elif(stsort[2][0]==seq_data['link'][2][0]):
                rfx_text += tm
            else:
                rfx_text += self.get_rfx_tr(tr)

            rfx_text += ori_text[stsort[2][1]:]
            
            seq_data['seq_rfx_tok']=self.t5_tokenizer.convert_tokens_to_ids(self.t5_tokenizer.tokenize(rfx_text))

    def get_elements_bert_tok_sted(self):
        for seq_data in self.seq_data_lis:
            char2tok_span = get_char2tok_spanlis_one_seq(seq_data['text'],self.tokenizer)
            seq_data['tm_tok_sted']=[char2tok_span[seq_data['link'][0][0]][0],char2tok_span[seq_data['link'][0][1]-1][1]-1]
            seq_data['tr_tok_sted']=[char2tok_span[seq_data['link'][1][0]][0],char2tok_span[seq_data['link'][1][1]-1][1]-1]
            seq_data['lg_tok_sted']=[char2tok_span[seq_data['link'][2][0]][0],char2tok_span[seq_data['link'][2][1]-1][1]-1]

    def get_graph(self):
        for seq_data in self.seq_data_lis:
            n_node = len(seq_data['node_to_tokid'])
            edg_matrix = seq_data['edgs']
            st_node = []
            ed_node = []
            for i in range(n_node):
                for j in range(n_node):
                    if(edg_matrix[i][j]>0):
                        st_node.append(i)
                        ed_node.append(j)
                        
            cg = dgl.graph((torch.tensor(st_node), torch.tensor(ed_node)),num_nodes=n_node)

            seq_data['graph'] = cg

    def collate_fn(self, batch):
        seq_bert_tok = [data['seq_bert_tok'] for data in batch]
        tm_tok_sted = [data['tm_tok_sted'] for data in batch]
        tr_tok_sted = [data['tr_tok_sted'] for data in batch]
        lg_tok_sted = [data['lg_tok_sted'] for data in batch]
        tm_nodelis = [data['tm_nodelis'] for data in batch]
        tr_nodelis = [data['tr_nodelis'] for data in batch]
        lg_nodelis = [data['lg_nodelis'] for data in batch]
        node_to_tokid = [data['node_to_tokid'] for data in batch]
        graph = batch[0]['graph']
        label = [data['label'] for data in batch]

        seq_t5_tok = [data['seq_t5_tok'] for data in batch]
        target_seq_t5_tok = [data['target_seq_t5_tok'] for data in batch]
        seq_rfx_tok = [data['seq_rfx_tok'] for data in batch]


        batch_size = len(seq_bert_tok)
        max_bertseq_len = max([len(s) for s in seq_bert_tok])
        batch_bert_data = [[0 for i in range(max_bertseq_len)]for j in range(batch_size)]
        batch_bert_mask_data = [[0 for i in range(max_bertseq_len)]for j in range(batch_size)]
        for j in range(batch_size):
            cur_len = len(seq_bert_tok[j])
            batch_bert_data[j][:cur_len] = seq_bert_tok[j]
            batch_bert_mask_data[j][:cur_len] = [1 for _ in range(cur_len)]

        batch_bert_data = torch.tensor(batch_bert_data, dtype=torch.long).to(self.device)
        batch_bert_mask_data = torch.tensor(batch_bert_mask_data, dtype=torch.long).to(self.device)
        batch_tm_tok_sted = torch.tensor(tm_tok_sted, dtype=torch.long).to(self.device)
        batch_tr_tok_sted = torch.tensor(tr_tok_sted, dtype=torch.long).to(self.device)
        batch_lg_tok_sted = torch.tensor(lg_tok_sted, dtype=torch.long).to(self.device)
        batch_tm_nodelis = torch.tensor(tm_nodelis, dtype=torch.long).to(self.device)
        batch_tr_nodelis = torch.tensor(tr_nodelis, dtype=torch.long).to(self.device)
        batch_lg_nodelis = torch.tensor(lg_nodelis, dtype=torch.long).to(self.device)
        batch_node_to_tokid = torch.tensor(node_to_tokid, dtype=torch.long).to(self.device)
        batch_graph = graph.to(self.device)
        batch_label = torch.tensor(label, dtype=torch.long).to(self.device)
        batch_t5_data = torch.tensor(seq_t5_tok, dtype=torch.long).to(self.device)
        batch_t5_target_data = torch.tensor(target_seq_t5_tok, dtype=torch.long).to(self.device)
        batch_seq_rfx_tok = torch.tensor(seq_rfx_tok, dtype=torch.long).to(self.device)

        return {
            'bert_data':batch_bert_data,
            'bert_mask_data':batch_bert_mask_data,
            'tm_tok_sted':batch_tm_tok_sted,
            'tr_tok_sted':batch_tr_tok_sted,
            'lg_tok_sted':batch_lg_tok_sted,
            'tm_nodelis':batch_tm_nodelis,
            'tr_nodelis':batch_tr_nodelis,
            'lg_nodelis':batch_lg_nodelis,
            'node_to_tokid':batch_node_to_tokid,
            'graph':batch_graph,
            'label': batch_label,
            't5_data':batch_t5_data,
            't5_target_data':batch_t5_target_data,
            'rfx_data': batch_seq_rfx_tok
        }

if __name__ == '__main__':
    seq_data_lis=[]
    import json
    from torch.utils.data import DataLoader
    
    with open('./data/json/HMCGR_train_data/45_N_22_E.jsonl','r') as f:
        for lin in f.readlines():
            seq_data_lis.append(json.loads(lin.strip()))
    test_dataset = HMCGRDataset(seq_data_lis)
    train_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=test_dataset.collate_fn)
    for i in train_loader:
        print(i)
        break

            


