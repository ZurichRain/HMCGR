import json
import os
import spacy
from transformers import BertTokenizer,BertTokenizerFast
from tqdm import tqdm
import neuralcoref
nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)
tokenizer = BertTokenizerFast.from_pretrained('/opt/data/private/sxu/fwang/transformers_model/bert-base-uncased')
            

elename_lis = ['PLACE','PATH','SPATIAL_ENTITY','NONMOTION_EVENT','MOTION','SPATIAL_SIGNAL','MOTION_SIGNAL','MEASURE']
linkname_lis = ['QSLINK','OLINK','MOVELINK']

def get_id_ele(eleid,data):
    for elename in elename_lis:
        for ele in data[elename.lower()]:
            if(ele['id']==eleid):
                return ele
    return None

def get_char2tok_spanlis_one_seq(seq,tokenizer):
    token_span = tokenizer.encode_plus(seq, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    char_num = None
    for tok_ind in range(len(token_span) - 1, -1, -1):
        if token_span[tok_ind][1] != 0:
            char_num = token_span[tok_ind][1]
            break
    char2tok_span = [[-1, -1] for _ in range(char_num)]
    for tok_ind, char_sp in enumerate(token_span):
        for char_ind in range(char_sp[0], char_sp[1]):
            tok_sp = char2tok_span[char_ind]
            if tok_sp[0] == -1:
                tok_sp[0] = tok_ind
            tok_sp[1] = tok_ind + 1
    return char2tok_span

def get_one_seq_graph_edgs(data):
    text = data['text']
    link = data['link']
    # print(text)
    # print(len(text))
    seqpass = nlp(text)
    words=[str(tok) for tok in list(seqpass)]
    ori_sted_lis = []
    cidx=0
    widx=0
    # print(words)
    while(cidx < len(text) and widx < len(words)):
        if text[cidx:cidx+len(words[widx])]==words[widx]:
            ori_sted_lis.append((cidx,cidx+len(words[widx])))
            cidx+=len(words[widx])-1
            widx+=1
        cidx+=1
    assert len(ori_sted_lis) == len(words)
    node_num = len(words)
    edgs = [[0 for j in range(node_num)] for i in range(node_num)]
    node_to_tokid,tm_nodelis,tr_nodelis,lg_nodelis = [],[],[],[]
    char2tok_span=get_char2tok_spanlis_one_seq(text,tokenizer)
    for idx,sted in enumerate(ori_sted_lis):
        st,ed = sted
        # print(st,':',ed)
        # print(char2tok_span[st])
        tokst,toked = char2tok_span[st][0], char2tok_span[ed-1][1]
        node_to_tokid.append([tokst,toked])
        e1st,e1ed = link[0][0],link[0][1]
        trst,tred = link[1][0],link[1][1]
        e2st,e2ed = link[2][0],link[2][1]
        if((e1st<ed and e1st >= st) or (e1ed>st and e1ed<=ed) or (e1st<=st and e1ed>=ed)):
            tm_nodelis.append(idx)
        if((trst<ed and trst >= st) or (tred>st and tred<=ed) or (trst<=st and tred>=ed)):
            tr_nodelis.append(idx)
        if((e2st<ed and e2st >= st) or (e2ed>st and e2ed<=ed) or (e2st<=st and e2ed>=ed)):
            lg_nodelis.append(idx)
    
    for elem in seqpass._.coref_clusters:
        core_span_lis = [[mention.start, mention.end] for mention in elem.mentions]
        for core_span1 in core_span_lis:
            for core_span2 in core_span_lis:
                if core_span1 == core_span2:
                    continue
                for i in range(core_span1[0],core_span1[1]):
                    for j in range(core_span2[0],core_span2[1]):
                        edgs[i][j]=1
    
    ele_set=set()
    for ele in tm_nodelis:
        ele_set.add(ele)
    for ele in tr_nodelis:
        ele_set.add(ele)
    for ele in lg_nodelis:
        ele_set.add(ele)
    for idx1 in ele_set:
        for idx2 in ele_set:
            if(idx1!=idx2):
                edgs[idx1][idx2]=1
    
    for idx,token in enumerate(seqpass):
        s,e = token.i, token.head.i
        edgs[s][e]=1
        edgs[e][s]=1
    
    return edgs,node_to_tokid,tm_nodelis,tr_nodelis,lg_nodelis
    

def get_one_doc_negative_examples(filename):
    seq_data_lis=[]
    with open(filename,'r') as f:
        for lin in f.readlines():
            seq_data_lis.append(json.loads(lin.strip()))
    all_link_data=[]
    # all_g,all_b=[],[]
    for data in seq_data_lis:
        # data['nolink'] = []
        # no_link_id = 0
        all_gold_link_lis = []
        all_bad_link_lis = []
        for linkname in linkname_lis:
            linktype = 1
            if linkname == 'OLINK':
                linktype=2
            elif linkname == 'MOVELINK':
                linktype=3
            tmele_lis=[]
            trele_lis=[]
            lgele_lis=[]

            cur_gold_link_lis=[]
            for lin in data[linkname.lower()]:
                tm = lin['trajector'] if linkname in ['QSLINK','OLINK'] else lin['mover']
                tr = lin['trigger']
                lg = lin['landmark'] if linkname in ['QSLINK','OLINK'] else lin['goal']
                # tmid,trid,lgid = tm,tr,lg
                if(tm != ''):
                    tmele = get_id_ele(tm,data)
                    # tmele_lis.append(tmele)
                    assert tmele['start']>=0 and tmele['end']<=len(data['text']) , print(tmele,len(data['text']))
                if(tr != ''):
                    trele = get_id_ele(tr,data)
                    # trele_lis.append(trele)
                    assert trele['start']>=0 and trele['end']<=len(data['text']) , print(trele,len(data['text']))
                if(lg != ''):
                    lgele = get_id_ele(lg,data)
                    # lgele_lis.append(lgele)
                    assert lgele['start']>=0 and lgele['end']<=len(data['text']) , print(lgele,len(data['text']))
                if(len(tm)>0 and len(tr)>0 and len(lg)>0):
                    tmele_lis.append(tmele)
                    trele_lis.append(trele)
                    lgele_lis.append(lgele)
                    cur_gold_link_lis.append((tm,tr,lg))
                    all_gold_link_lis.append([(tmele['start'],tmele['end']),(trele['start'],trele['end']),\
                                            (lgele['start'],lgele['end']),linktype])
            
            memory_bad_link = []
            for e1 in tmele_lis:
                for ctr in trele_lis:
                    for e2 in lgele_lis:
                        if e1['id'] == e2['id']:
                            continue
                        if((e1['id'],ctr['id'],e2['id']) in cur_gold_link_lis or (e1['id'],ctr['id'],e2['id']) in memory_bad_link):
                            continue
                        # c_bad_dict=dict()
                        # c_bad_dict['id']= 'no'+str(no_link_id)
                        # no_link_id+=1
                        # c_bad_dict['trajector'] = e1['id']
                        # c_bad_dict['trigger'] = ctr['id']
                        # c_bad_dict['landmark'] = e2['id']
                        # data['nolink'].append(c_bad_dict)
                        tmele = get_id_ele(e1['id'],data)
                        trele = get_id_ele(ctr['id'],data)
                        lgele = get_id_ele(e2['id'],data)
                        all_bad_link_lis.append([(tmele['start'],tmele['end']),(trele['start'],trele['end']),\
                                            (lgele['start'],lgele['end']),0])
                        memory_bad_link.append((e1['id'],ctr['id'],e2['id']))
        # all_gold_link  
        # all_bad_link
        link_data_lis=[]
        for g_link in all_gold_link_lis:
            link_data_dict = dict()
            link_data_dict['text'] = data['text']
            link_data_dict['link'] = g_link
            link_data_lis.append(link_data_dict)
        for b_link in all_bad_link_lis:
            link_data_dict = dict()
            link_data_dict['text'] = data['text']
            link_data_dict['link'] = b_link
            link_data_lis.append(link_data_dict)


        for link_data in link_data_lis:
            # get_one_seq_graph_edgs(link_data)
            edgs, node_to_tokid,tm_nodelis,tr_nodelis,lg_nodelis = get_one_seq_graph_edgs(link_data)
            link_data['edgs'] = edgs
            link_data['node_to_tokid'] = node_to_tokid
            link_data['tm_nodelis'] = tm_nodelis
            link_data['tr_nodelis'] = tr_nodelis
            link_data['lg_nodelis'] = lg_nodelis
        
        all_link_data+=link_data_lis  

    return all_link_data  

def get_dir_negative_examples(dirname,model_data_dirname):

    for file in tqdm(os.listdir(dirname)):
        if(file.split('.')[-1] != 'jsonl'):
            continue
        all_link_data = get_one_doc_negative_examples(os.path.join(dirname,file))

        # g,b=get_one_doc_negative_examples(os.path.join(dirname,file))
        # all_g+=g
        # all_b+=b
        json_file = os.path.join(model_data_dirname,file)
        with open(json_file,'w') as f:
            for link_data in all_link_data:
                json_str=json.dumps(link_data,ensure_ascii=False)
                f.write(json_str+'\n')



if __name__ == '__main__':
    train_clear_seq_dir = './data/json/clear_seq_train_data'
    train_model_data_dir = './data/json/HMCGR_train_data'
    get_dir_negative_examples(train_clear_seq_dir,train_model_data_dir)

    vail_clear_seq_dir = './data/json/clear_seq_vail_data'
    vail_model_data_dir = './data/json/HMCGR_vail_data'
    get_dir_negative_examples(vail_clear_seq_dir,vail_model_data_dir)

    test_clear_seq_dir = './data/json/clear_seq_test_data'
    test_model_data_dir = './data/json/HMCGR_test_data'
    get_dir_negative_examples(test_clear_seq_dir,test_model_data_dir)
