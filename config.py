'''
HMCGR config
'''
import os
import torch

model_name= 'HMCGR'
sub_model_name = 'joint_three_link'
device='cuda:0' if torch.cuda.is_available() else 'cpu'

do_train = True
do_test = True
use_tqdm = False
test_gold = True
k_fold = False


epoch_num = 30 
min_epoch_num = 3
learning_rate = 1e-5
batch_size= 1

clip_grad = 5 
load_before = False
patience = 0.0002 
patience_num = 50 
full_fine_tuning=True 
weight_decay=0.1


model_type = model_name+'_'+sub_model_name

model_parameter_adjustment_name=model_name+'_'+sub_model_name+'_lr_2e_5_bz_'+str(batch_size)
model_parameter_adjustment_eval_result_name=model_parameter_adjustment_name+'_log'

bert_pathname = "/opt/data/private/sxu/fwang/transformers_model/bert-base-uncased/"
longformer_pathname = 'allenai/longformer-base-4096'
T5base_pathname='/opt/data/private/sxu/fwang/transformers_model/T5_base/'

train_data_dir='./data/json/HMCGR_train_data/'
vail_data_dir='./data/json/HMCGR_vail_data/'
test_data_dir='./data/json/HMCGR_test_data/' 

train_log_dir='./log/'+model_name+'_log/'+sub_model_name+'/train_log/'
test_log_dir='./log/'+model_name+'_log/'+sub_model_name+'/test_log/'



if k_fold:
    save_train_model_dir='./k_fold_model/'+model_name+'_model/'+sub_model_name+'/'
else:
    save_train_model_dir='./save_model/'+model_name+'_model/'+sub_model_name+'/'
save_train_model_file=save_train_model_dir+model_parameter_adjustment_name

if k_fold :
    save_train_result_dir='./vail_test_k_fold_data_result/'+model_name+'_model/'+sub_model_name+'/'
else:
    save_train_result_dir='./vail_test_data_result/'+model_name+'_model/'+sub_model_name+'/'



save_train_result_file=save_train_result_dir+model_parameter_adjustment_eval_result_name

save_finall_result_dir='./final_result/'+model_name+'_model/'+sub_model_name+'/'
save_finall_result_file=save_finall_result_dir+model_parameter_adjustment_eval_result_name

save_test_all_res_dir='./all_res/'+model_name+'_model/'+sub_model_name+'/'
save_res_file = save_test_all_res_dir + model_parameter_adjustment_eval_result_name





ele_labels=['spatial_entity','nonmotion_event','motion','spatial_signal','motion_signal','measure','place','path']
ele_lis_name=['place_lis','path_lis','spatial_entity_lis','motion_lis','spatial_signal_lis','motion_signal_lis',
                    'nonmotion_event_lis','measure_lis']
ele_attributes_name={
    'place_lis':['type','dimensionality','form','domain','continent','state','country','ctv','gazref','latlong','elevation','mod',
                'dcl','countable','gquat','scopes'],
    'path_lis':[ 'beginid','endid','midid','type','dimensionality','form','domain','gazref','latlong','elevation','mod','dcl',
                'countable','gquat','scopes'],
    'spatial_entity_lis':['type','dimensionality','form','domain','latlong','elevation','mod','dcl','countable','gquat','scopes'],
    'motion_lis':['domain','latlong','elevation','motion_type','motion_class','motion_sense','mod','countable','gquant','scopes'],
    'spatial_signal_lis':['cluster','semantic_type'],
    'motion_signal_lis':['motion_signal_type'],
    'nonmotion_event_lis':['domain','latlong','elevation','mod','countable','gquant','scopes'],
    'measure_lis':['value','unit']
}

link_lis_name=['qslink_lis','olink_lis','movelink_lis']
link_attributes_name={
    'qslink_lis':['reltype','trajector','landmark','trigger'],
    'olink_lis':['reltype','trajector','landmark','trigger','frame_type','referencept','projective'],
    'movelink_lis':['trigger','source','goal','midpoint','mover','landmark','goal_reached','pathid','motion_signalid']
}
spatial_qsrole_label2id={
    'O':0,
    'B-qstj':1,
    'I-qstj':2,
    'B-qsld':3,
    'I-qsld':4,
    'B-qstr':5,
    'I-qstr':6
}

spatial_qsorole_label2id={
    'O':0,
    'B-qstj':1,
    'I-qstj':2,
    'B-qsld':3,
    'I-qsld':4,
    'B-qstr':5,
    'I-qstr':6,
    'B-otj':7,
    'I-otj':8,
    'B-old':9,
    'I-old':10,
    'B-otr':11,
    'I-otr':12
}

spatial_qsomvrole_label2id={
    'O':0,
    'B-qstj':1,
    'I-qstj':2,
    'B-qsld':3,
    'I-qsld':4,
    'B-qstr':5,
    'I-qstr':6,
    'B-otj':7,
    'I-otj':8,
    'B-old':9,
    'I-old':10,
    'B-otr':11,
    'I-otr':12,
    'B-mvmo':13,
    'I-mvmo':14,
    'B-mvgo':15,
    'I-mvgo':16,
    'B-mvtr':17,
    'I-mvtr':18
}


spatial_ele_label2id={
    'O':0,
    'B-spatial_entity':1,
    'I-spatial_entity':2,
    'B-nonmotion_event':3,
    'I-nonmotion_event':4,
    'B-motion':5,
    'I-motion':6,
    'B-spatial_signal':7,
    'I-spatial_signal':8,
    'B-motion_signal':9,
    'I-motion_signal':10,
    'B-measure':11,
    'I-measure':12,
    'B-place':13,
    'I-place':14,
    'B-path':15,
    'I-path':16
}


spatial_role_label2id={
    'O':0,
    'B-trajector':1,
    'I-trajector':2,
    'B-landmark':3,
    'I-lanmark':4,
    'B-trigger':5,
    'I-trigger':6,
    'B-mover':7,
    'I-mover':8,
    'B-goal':9,
    'I-goal':10
}
spatial_role_label2id1={
    'O':0,
    'B-qstj':1,
    'I-qsth':2,
    'B-qsld':3,
    'I-qsld':4,
    'B-qstr':5,
    'I-qstr':6,
    'B-otj':7,
    'I-otj':8,
    'B-old':9,
    'I-old':10,
    'B-otr':11,
    'I-otr':12,
    'B-mv':13,
    'I-mv':14,
    'B-gl':15,
    'I-gl':16,
    'B-mo':17,
    'I-mo':18,
    'B-metj':19,
    'I-metj':20,
    'B-meld':21,
    'I-meld':22,
    'B-metr':23, 
    'I-metr':24
}
