import os
import sys
import xml.dom.minidom as xmd
import json

def init_dict(adict):
    adict['text'] = ''
    adict['place'] = []
    adict['path'] = []
    adict['spatial_entity'] = []
    adict['nonmotion_event'] = []
    adict['motion'] = []
    adict['spatial_signal'] = []
    adict['motion_signal'] = []
    adict['measure'] = []

    adict['qslink'] = []
    adict['olink'] = []
    adict['movelink'] = []

def get_one_type_elements(dom,elename):
    res_lis=[]
    measures = dom.getElementsByTagName(elename)
    for p in measures:
        cur_dict = dict()
        for k,v in p.attributes.items():
            curk=k.lower()
            cur_dict[curk]=v
        res_lis.append(cur_dict)
    return res_lis


def convert_xml_to_json(xml_file):
    res_dict=dict()
    init_dict(res_dict)
    dom=xmd.parse(xml_file)
    root = dom.documentElement

    texts = root.getElementsByTagName('TEXT')
    for text in texts:  
        for child in text.childNodes:
            res_dict['text']+=child.data
            res_dict['text']+=' '

    elename_lis = ['PLACE','PATH','SPATIAL_ENTITY','NONMOTION_EVENT','MOTION','SPATIAL_SIGNAL','MOTION_SIGNAL','MEASURE']
    for elename in elename_lis:
        res_dict[elename.lower()] = get_one_type_elements(dom,elename)
    
    linkname_lis = ['QSLINK','OLINK','MOVELINK']
    for linkname in linkname_lis:
        res_dict[linkname.lower()] = get_one_type_elements(dom,linkname)
    
    return res_dict

def convert_dir_xml_to_json(xml_dir,json_dir):
    for xml_file in os.listdir(xml_dir):
        if(xml_file.split('.')[-1] != 'xml'):
            continue
        
        json_dict = convert_xml_to_json(os.path.join(xml_dir,xml_file))
        json_file = os.path.join(json_dir,xml_file.split('.')[0]+'.json')
        with open(json_file,'w') as f:
            json_str=json.dumps(json_dict,ensure_ascii=False)
            f.write(json_str)

if __name__ == '__main__':
    train_xml_dir = './data/xml/train_data'
    train_json_dir= './data/json/train_data'
    # convert_dir_xml_to_json(train_xml_dir,train_json_dir)
    vail_xml_dir = './data/xml/vail_data'
    vail_json_dir= './data/json/vail_data'
    # convert_dir_xml_to_json(vail_xml_dir,vail_json_dir)
    test_xml_dir = './data/xml/test_data'
    test_json_dir= './data/json/test_data'
    # convert_dir_xml_to_json(test_xml_dir,test_json_dir)
        
    # with open('data/json/test_data/48_N_8_E.json','r') as f:
    #     data = json.load(f)
    # print(data['qslink'])
