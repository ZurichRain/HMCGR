from util import remove_extra_spaces,elename_lis,linkname_lis
import json
import os

def vail_one_attribute(data,attr,ntext,oriidx2newidx):
    new_lis=[]
    for p in data[attr]:
        if(int(p['start'])==-1):
            cdict = dict()
            cdict['id'] = p['id']
            cdict['new_text'] = p['text']
            cdict['new_start'] = -1
            cdict['new_end'] = -1
            new_lis.append(cdict)
            continue
        

        assert [w for w in ntext[oriidx2newidx[int(p['start'])]:oriidx2newidx[int(p['end'])]].split(' ') if len(w)>0] == \
            [w for w in p['text'].split(' ') if len(w)>0]
        p['new_start'] = oriidx2newidx[int(p['start'])]
        p['new_end'] = oriidx2newidx[int(p['end'])]
        p['new_text'] = ntext[oriidx2newidx[int(p['start'])]:oriidx2newidx[int(p['end'])]]
        cdict = dict()
        cdict['id'] = p['id']
        cdict['new_text'] = p['new_text']
        cdict['new_start'] = p['new_start']
        cdict['new_end'] = p['new_end']
        new_lis.append(cdict)
    return new_lis


def clear_one_json_data(file):
    with open(file,'r') as f:
        data = json.load(f)
    use_data=dict()

    ntext,oriidx2newidx=remove_extra_spaces(data['text'])
    use_data['text'] = ntext
    for ele in elename_lis:
        use_data[ele.lower()]=vail_one_attribute(data,ele.lower(),ntext,oriidx2newidx)
    for linkname in linkname_lis:
        linkname = linkname.lower()
        use_lin_lis = []
        for lin in data[linkname]:
            cdict=dict()
            cdict['id'] = lin['id']
            if linkname in ['qslink','olink']:
                cdict['trajector'] = lin['trajector']
                cdict['trigger'] = lin['trigger']
                cdict['landmark'] = lin['landmark']
            else:
                cdict['mover'] = lin['mover']
                cdict['trigger'] = lin['trigger']
                cdict['goal'] = lin['goal']
            use_lin_lis.append(cdict)
        use_data[linkname.lower()] = use_lin_lis
    return use_data
    
def clear_one_dir(dir,clear_dir):
    for file in os.listdir(dir):
        if(file.split('.')[-1] != 'json'):
            continue
        clear_data=clear_one_json_data(os.path.join(dir,file))
        json_file = os.path.join(clear_dir,file)
        with open(json_file,'w') as f:
            json_str=json.dumps(clear_data,ensure_ascii=False)
            f.write(json_str)

if __name__ == '__main__':
    train_json_dir= './data/json/train_data'
    train_clear_json_dir = './data/json/clear_train_data'
    clear_one_dir(train_json_dir,train_clear_json_dir)
    vail_json_dir= './data/json/vail_data'
    vail_clear_json_dir = './data/json/clear_vail_data'
    clear_one_dir(vail_json_dir,vail_clear_json_dir)
    test_json_dir= './data/json/test_data'
    test_clear_json_dir = './data/json/clear_test_data'
    clear_one_dir(test_json_dir,test_clear_json_dir)
    

    