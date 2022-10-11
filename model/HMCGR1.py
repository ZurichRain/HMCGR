from transformers import BertTokenizer,BertModel,T5ForConditionalGeneration
import torch.nn as nn
import torch
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
import config
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F

class HMCGR(nn.Module):
    def __init__(self):
        super(HMCGR,self).__init__()
        self.num_label = 4
        self.bert = BertModel.from_pretrained(config.bert_pathname)
        self.T5c=T5ForConditionalGeneration.from_pretrained(config.T5base_pathname)
        self.tanh=nn.Tanh()
        self.outlin=nn.Linear(12*self.bert.config.hidden_size,self.num_label)
        torch.nn.init.xavier_uniform_(self.outlin.weight)
        self.ce_loss = nn.CrossEntropyLoss()

        self.span_extractor=SelfAttentiveSpanExtractor(input_dim=2*self.bert.config.hidden_size)
        self.span_extractor_for_dgl=SelfAttentiveSpanExtractor(input_dim=self.bert.config.hidden_size)
        self.conv1 = GraphConv(2*self.bert.config.hidden_size, 2*768, weight=True, allow_zero_in_degree=True, norm='right')
        
    def get_attention(self,bert_emb,t5_encode_emb):
        score=torch.bmm(bert_emb,t5_encode_emb.transpose(-2,-1))
        prob = F.softmax(score,-1)
        context=torch.bmm(prob,t5_encode_emb)
        cat_emb=torch.cat((bert_emb,context),dim=-1)
        return cat_emb

    
    def get_sum_list_emb(self,emb_lis):
        return torch.mean(torch.cat(emb_lis,dim=0),dim=0)

    def forward(self, bert_data,bert_mask_data,tm_tok_sted,tr_tok_sted,lg_tok_sted,\
                tm_nodelis,tr_nodelis,lg_nodelis,node_to_tokid,graph,label,\
                t5_data,t5_target_data,rfx_data):
        
        bertout = self.bert(input_ids=bert_data,attention_mask=bert_mask_data)
        wemb = bertout[0]
        outputs = self.T5c(input_ids=t5_data, labels=t5_target_data)
        t5_loss = outputs.loss
        encoder = self.T5c.get_encoder()
        encoder_outputs = encoder(
            input_ids=t5_data,
            attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        rfx_encoder_outputs = encoder(
            input_ids=rfx_data,
            attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
        )
        encoder_hidden_states = encoder_outputs[0]
        rfx_hidden_states = rfx_encoder_outputs[0]

        ori_emb = torch.mean(encoder_hidden_states,dim=1)
        rfx_emb = torch.mean(rfx_hidden_states,dim=1)

        rfx_loss = 1-torch.cosine_similarity(ori_emb,rfx_emb)
        rfx_loss = rfx_loss.squeeze(0)
        
        wemb=self.get_attention(wemb,encoder_hidden_states)

        tm_emb=self.span_extractor(wemb,tm_tok_sted.unsqueeze(1))
        tr_emb=self.span_extractor(wemb,tr_tok_sted.unsqueeze(1))
        lg_emb=self.span_extractor(wemb,lg_tok_sted.unsqueeze(1))

        node_emb = self.span_extractor(wemb,node_to_tokid)[0]

        
        
        h1=self.conv1(graph,node_emb)
        cur_lis=[]
        for idx in tm_nodelis:
            cur_lis.append(h1[idx])
        tm_node_emb = self.get_sum_list_emb(cur_lis)
        
        cur_lis=[]
        for idx in tr_nodelis:
            cur_lis.append(h1[idx])
        tr_node_emb = self.get_sum_list_emb(cur_lis)
        
        for idx in lg_nodelis:
            cur_lis.append(h1[idx])
        lg_node_emb = self.get_sum_list_emb(cur_lis)

        tm_emb = tm_emb[0]+tm_node_emb
        tr_emb = tr_emb[0]+tr_node_emb
        lg_emb = lg_emb[0]+lg_node_emb
        
        activate_emb=torch.cat((tm_emb,tr_emb,lg_emb,torch.abs(tm_emb-tr_emb),torch.abs(lg_emb-tr_emb),torch.abs(lg_emb-tm_emb)),dim=-1)
        activate_emb = self.outlin(activate_emb)
        
        logits = activate_emb.view(-1,self.num_label)
        ctrain_y = label.view(-1)
        l1=self.ce_loss(logits,ctrain_y)
        return l1+rfx_loss+t5_loss,logits