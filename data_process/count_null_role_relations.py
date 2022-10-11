import json
import os

def count_null_role_relations_one_doc(filename):
    data_lis=[]
    with open(filename,'r') as f:
        for lin in f.readlines():
            # print(lin)
            data_lis.append(json.loads(lin.strip()))
    # print(data_lis)
    null_num=0
    all_role_num=0
    for data in data_lis:
        for lin in data['qslink']:
            if lin['trajector'] == '' or lin['trigger'] == '' or lin['landmark'] == '':
                null_num+=1
            else:
                all_role_num+=1

        for lin in data['olink']:
            if lin['trajector'] == '' or lin['trigger'] == '' or lin['landmark'] == '':
                null_num+=1
            else:
                all_role_num+=1

        for lin in data['movelink']:
            if lin['mover'] == '' or lin['trigger'] == '' or lin['goal'] == '':
                null_num+=1
            else:
                all_role_num+=1
    
    return null_num,all_role_num

def count_null_role_relations_dir(dirname):
    all_null_role_n = 0
    all_role_n = 0
    for file in os.listdir(dirname):
        if(file.split('.')[-1] != 'jsonl'):
            continue
        null_num,all_role_num=count_null_role_relations_one_doc(os.path.join(dirname,file))
        all_null_role_n+=null_num
        all_role_n += all_role_num
    print(all_null_role_n,':', all_role_n)
    print('null pecentage:{}%'.format(all_null_role_n*1.0/(all_null_role_n+all_role_n)*100))
    print('all pecentage:{}%'.format(all_role_n*1.0/(all_null_role_n+all_role_n)*100))
    
if __name__ == '__main__':
    train_clear_seq_dir = './data/json/clear_seq_train_data'
    count_null_role_relations_dir(train_clear_seq_dir)

    vail_clear_seq_dir = './data/json/clear_seq_vail_data'
    count_null_role_relations_dir(vail_clear_seq_dir)

    test_clear_seq_dir = './data/json/clear_seq_test_data'
    count_null_role_relations_dir(test_clear_seq_dir)