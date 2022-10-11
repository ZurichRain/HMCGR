import json
import os
elename_lis = ['PLACE','PATH','SPATIAL_ENTITY','NONMOTION_EVENT','MOTION','SPATIAL_SIGNAL','MOTION_SIGNAL','MEASURE']
linkname_lis = ['QSLINK','OLINK','MOVELINK']

def get_ele_by_id(data,eleid):
    for elename in elename_lis:
        for ele in data[elename.lower()]:
            if(ele['id']==eleid):
                return ele
    return None

def get_qslink_olink_one_doc(filename):
    data_lis=[]
    with open(filename,'r') as f:
        for lin in f.readlines():
            data_lis.append(json.loads(lin.strip()))

    qslink_lis = []
    olink_lis = []
    for data in data_lis:
        for qslin in data['qslink']:
            tm = ''
            if len(qslin['trajector'])>0:
                tm = get_ele_by_id(data,qslin['trajector'])['text']
            tr = ''
            if len(qslin['trigger'])>0:
                tr = get_ele_by_id(data,qslin['trigger'])['text']
            lg = ''
            if len(qslin['landmark'])>0:
                lg = get_ele_by_id(data,qslin['landmark'])['text']
            qslink_lis.append([tm,tr,lg])
        for olin in data['olink']:
            tm = ''
            if len(olin['trajector'])>0:
                tm = get_ele_by_id(data,olin['trajector'])['text']
            tr = ''
            if len(olin['trigger'])>0:
                tr = get_ele_by_id(data,olin['trigger'])['text']
            lg = ''
            if len(olin['landmark'])>0:
                lg = get_ele_by_id(data,olin['landmark'])['text']
            olink_lis.append([tm,tr,lg])
    
    return qslink_lis,olink_lis

def get_qslink_olink_all_dir(dirname):
    all_qslink_lis = []
    all_olink_lis = []
    for file in os.listdir(dirname):
        if(file.split('.')[-1] != 'jsonl'):
            continue
        qslink_lis,olink_lis=get_qslink_olink_one_doc(os.path.join(dirname,file))
        all_qslink_lis += qslink_lis
        all_olink_lis += olink_lis
    print('qslink_num:{}'.format(len(all_qslink_lis)))
    print(all_qslink_lis[:5])
    print('olink_num:{}'.format(len(all_olink_lis)))
    print(all_olink_lis[:5])
    
if __name__ == '__main__':
    train_clear_seq_dir = './data/json/clear_seq_train_data'
    get_qslink_olink_all_dir(train_clear_seq_dir)

    vail_clear_seq_dir = './data/json/clear_seq_vail_data'
    get_qslink_olink_all_dir(vail_clear_seq_dir)

    test_clear_seq_dir = './data/json/clear_seq_test_data'
    get_qslink_olink_all_dir(test_clear_seq_dir)