import os
import json
from util import elename_lis,linkname_lis
from nltk.tokenize import sent_tokenize
import copy
# import nltk
# nltk.download('punkt')


def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list

def my_sentence_lis_split(text):
    endsig=['.','!','?']
    stchar=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    def check(alltext,cid,stchar,endsig):
        c=cid
        if(alltext[c]=='S' or alltext[c]=='A'):
            return False
        while(c<len(alltext)):
            if(alltext[c]==' ' or alltext[c] =='\n' or alltext[c] in endsig):
                c+=1
            elif(alltext[c] in stchar):
                return True
            elif(alltext[c] not in stchar):
                return False
    curseqlis = []
    curseqtxt=''
    seqid=0
    for cid in range(len(text)):
        if((text[cid-1] in endsig and text[cid]==' ') and check(text,cid,stchar,endsig)):
            continue
        if(text[cid] in endsig and check(text,cid+1,stchar,endsig)):
            
            curseqtxt+=text[cid]
            curseqlis.append(curseqtxt)
            
            curseqtxt=''
            seqid+=1
            continue
        curseqtxt+=text[cid]
    curseqlis.append(curseqtxt)
    return curseqlis

def get_one_seq_ele(seq_dict,ele_name,ele_lis,seq_st,seq_ed):
    seq_dict[ele_name]=[]
    cur_id_lis = []
    for e in ele_lis:
        if(e['new_start']>=seq_st and e['new_start']<seq_ed):
            ce = dict()
            ce['id'] = e['id']
            ce['text'] = e['new_text']
            ce['start'] = e['new_start']-seq_st
            ce['end'] = e['new_end']-seq_st
            assert seq_dict['text'][ce['start']:ce['end']] == e['new_text'],print(e)
            seq_dict[ele_name].append(ce)
            cur_id_lis.append(e['id'])
    return cur_id_lis

def get_one_seq_link(seq_dict,link_name,link_lis,seq_ele_id_lis):
    seq_dict[link_name]=[]
    e1_name = 'trajector' if link_name in ['qslink','olink'] else 'mover'
    tr_name = 'trigger'
    e2_name = 'landmark' if link_name in ['qslink','olink'] else 'goal'
    for lin in link_lis:
        if(lin[e1_name] in seq_ele_id_lis or lin[tr_name] in seq_ele_id_lis or lin[e2_name] in seq_ele_id_lis):
            if(lin[e1_name] != '' and lin[e1_name] not in seq_ele_id_lis):
                continue
            if(lin[tr_name] != '' and lin[tr_name] not in seq_ele_id_lis):
                continue
            if(lin[e2_name] != '' and lin[e2_name] not in seq_ele_id_lis):
                continue
            seq_dict[link_name].append(lin)
    

def split_one_doc_to_seqs(filename):

    with open(filename,'r') as f:
        data = json.load(f)
    
    seqlis = my_sentence_lis_split(data['text'])
    seq_st_ed=[]
    suffix_len = 0
    one_doc_seq_data_lis = []
    for seq in seqlis:
        seq_st_ed.append((suffix_len,suffix_len+len(seq)))
        suffix_len+=len(seq)+1
        assert seq == data['text'][seq_st_ed[-1][0]:seq_st_ed[-1][1]]
        seq_dict=dict()
        seq_dict['text'] = seq
        cur_one_seq_all_ele_id_lis=[]
        for elename in elename_lis:
            cur_one_seq_all_ele_id_lis += get_one_seq_ele(seq_dict,elename.lower(),data[elename.lower()],seq_st_ed[-1][0],seq_st_ed[-1][1])
        for linkname in linkname_lis:
            get_one_seq_link(seq_dict,linkname.lower(),data[linkname.lower()],cur_one_seq_all_ele_id_lis)
        
        one_doc_seq_data_lis.append(seq_dict)
    

    return one_doc_seq_data_lis
    
    
def split_dir_doc_to_seqs(dirname,seq_dirname):
    for file in os.listdir(dirname):
        if(file.split('.')[-1] != 'json'):
            continue
        one_doc_seq_data_lis=split_one_doc_to_seqs(os.path.join(dirname,file))
        json_file = os.path.join(seq_dirname,file+'l')
        with open(json_file,'w') as f:
            for seq_data in one_doc_seq_data_lis:
                json_str=json.dumps(seq_data,ensure_ascii=False)
                f.write(json_str+'\n')

if __name__ == '__main__':
    train_clear_json_dir = './data/json/clear_train_data'
    train_clear_seq_dir = './data/json/clear_seq_train_data'
    split_dir_doc_to_seqs(train_clear_json_dir,train_clear_seq_dir)


    vail_clear_json_dir = './data/json/clear_vail_data'
    vail_clear_seq_dir = './data/json/clear_seq_vail_data'
    split_dir_doc_to_seqs(vail_clear_json_dir,vail_clear_seq_dir)


    test_clear_json_dir = './data/json/clear_test_data'
    test_clear_seq_dir = './data/json/clear_seq_test_data'
    split_dir_doc_to_seqs(test_clear_json_dir,test_clear_seq_dir)


    

