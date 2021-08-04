import torch
from importlib import import_module
import os
from utils import *
import jieba.posseg as psg
import json
import codecs
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stdout.write("Your content....")

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
key = {
    0: 'finance',
    1: 'realty',
    2: 'stocks',
    3: 'education',
    4: 'science',
    5: 'society',
    6: 'politics',
    7: 'sports',
    8: 'game',
    9: 'entertainment'
}
#print(ployphone_ch_id)
model_name = 'bert_haitian_CNN_mask'#'bert_haitian_CNN'#'bert_CNN'
x = import_module('models.' + model_name)
config = x.Config('haitian_data/data_100')
model = x.Model(config).to(config.device)
print('path ',config.save_path+'\n\n')
model.load_state_dict(torch.load(config.save_path, map_location='cpu'))
pos_dict = json.load(open('pos.json'))

def build_predict_text(text,position_ployphone,sentence_pos,segment):
    mask_windows=config.windows
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
            sentence_pos = sentence_pos+[0]*(pad_size-len(sentence_pos)) #词性
            segment = segment+[0]*(pad_size-len(segment))#分词
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    
    #jiang
    #mask = [0]*pad_size #jiang
    seq_len = pad_size
    if mask_windows:
        mask = [0]*pad_size
        if len(token)<mask_windows:
        
            for index_mask in range(0,len(token)):
            
                mask[index_mask]=1
        elif int(position)-mask_windows//2<0:
            for index_mask in range(0,mask_windows):
                mask[index_mask]=1
        elif int(position)+mask_windows//2+1>len(token):
            for index_mask in range(1,mask_windows+1):
                if len(token)>50:
                    mask[-index_mask]=1
                else:
                    mask[len(token)-index_mask]=1
        else:
        
            for index_mask in range(1,mask_windows//2+1):
                mask[int(position)-1+index_mask]=1
                mask[int(position)-1-index_mask]=1
            mask[int(position)]=1

    '''
    ploymasked = [0]*len(ployphone_id) #jiang
    for m in ployphone_ch_id[ployphone]: #jiang
        ploymasked[ployphone_id[m]]=1#jiang
    positions = [0.0]*len(token_ids)#jiang
    
    positions[int(position)]=1.0#jiang
    #return [token_ids,seq_len,mask,ploymasked,positions,sentence_pos,segment]
    #'''
    ploymasked_list =[]
    position_list=[]
    for position_,ployphone_ in position_ployphone:
        ploymasked = [0]*len(ployphone_id) #jiang
        for m in ployphone_ch_id[ployphone_]:
            ploymasked[ployphone_id[m]]=1#jiang
        #positions = [0.0]*len(token_ids)#jiang
        #positions[int(position_)]=1.0#jiang
        positions = [int(position_)+1]
        ploymasked_list.append(ploymasked)
        position_list.append(positions)
    
    #jiang
    ids = torch.LongTensor([token_ids]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    ploymasked = torch.LongTensor([ploymasked_list]).to(config.device)
    positions = torch.LongTensor([position_list]).to(config.device)
    sentence_pos = torch.LongTensor([sentence_pos]).to(config.device)
    segment = torch.LongTensor([segment]).to(config.device)
    return ids, seq_len, mask,ploymasked,positions,sentence_pos,segment
    #'''
def build_predict_text2(text,position,ployphone,sentence_pos,segment):
    mask_windows=config.windows
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + ([0] * (pad_size - len(token)))
            token_ids += ([0] * (pad_size - len(token)))
            sentence_pos = sentence_pos+[0]*(pad_size-len(sentence_pos)) #词性
            segment = segment+[0]*(pad_size-len(segment))#分词
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    
    #jiang
    #mask = [0]*pad_size #jiang
    #seq_len = pad_size
    if mask_windows:
        mask = [0]*pad_size
        if len(token)<mask_windows:
        
            for index_mask in range(0,len(token)):
            
                mask[index_mask]=1
        elif int(position)-mask_windows//2<0:
            for index_mask in range(0,mask_windows):
                mask[index_mask]=1
        elif int(position)+mask_windows//2+1>len(token):
            for index_mask in range(1,mask_windows+1):
                if len(token)>50:
                    mask[-index_mask]=1
                else:
                    mask[len(token)-index_mask]=1
        else:
        
            for index_mask in range(1,mask_windows//2+1):
                mask[int(position)-1+index_mask]=1
                mask[int(position)-1-index_mask]=1
            mask[int(position)]=1

    #'''
    ploymasked = [0]*len(ployphone_id) #jiang
    for m in ployphone_ch_id[ployphone]: #jiang
        ploymasked[ployphone_id[m]]=1#jiang
    positions = [int(position)+1]
    '''
    positions = [0.0]*len(token_ids)#jiang
    print('position ',position) 
    positions[int(position)]=1.0#jiang
    #return [token_ids,seq_len,mask,ploymasked,positions,sentence_pos,segment]
    
    ploymasked_list =[]
    position_list=[]
    for position_,ployphone_ in position_ployphone:
        ploymasked = [0]*len(ployphone_id) #jiang
        for m in ployphone_ch_id[ployphone_]:
            ploymasked[ployphone_id[m]]=1#jiang
        positions = [0.0]*len(token_ids)#jiang
        positions[int(position_)]=1.0#jiang
        ploymasked_list.append(ploymasked)
        position_list.append(positions)
    '''
    #jiang
    ids = torch.LongTensor([token_ids]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)
    ploymasked = torch.LongTensor([ploymasked]).to(config.device)
    positions = torch.LongTensor([positions]).to(config.device)
    sentence_pos = torch.LongTensor([sentence_pos]).to(config.device)
    segment = torch.LongTensor([segment]).to(config.device)
    return ids, seq_len, mask,ploymasked,positions,sentence_pos,segment

def get_chinese_segment(chinese:str):
    '''
    将汉语进行分词，获得分词的词性和分词的结果
    分词：0起始符号，1开始，1为中间，3为结尾
    :param chinese:
    :return:
    '''
    CWS=[] # 分词
    POS=[] #词性
    jieba_cut = psg.cut(chinese)
    for w in jieba_cut:
        word = w.word
        flag = w.flag
        if not pos_dict.__contains__(flag):
            pos_dict[flag]=len(pos_dict)
        len_word = len(word)
        if len_word>2:
            CWS_temp = [2]*len_word
            CWS_temp[1],CWS_temp[-1]=1,3
            CWS.extend(CWS_temp)
        elif len_word==2:
            CWS.extend([1,3])
        else:
            CWS.append(1)
        POS.extend([pos_dict[flag]]*len_word)
    if len(chinese)!=len(POS):
        print(chinese)
    return CWS,POS

def predict(text,sentence_pos,segment,position=None,ployphone=None):
    """
    单个文本预测
    position 为列表
    :param text:
    :return:
    """
    #ployphone = ployphone.encode('utf-8', errors='surrogateescape').decode('utf-8')
    #print('input',position,ployphone)
    if position==None and ployphone!=None:
        position =[ (i,ployphone) for i,s in enumerate(text) if s==ployphone]
    if position==None and ployphone==None:
        position_ployphone = [(i,s) for i,s in enumerate(text) if ployphone_ch_id.__contains__(s)]
    print('position ',len(position_ployphone))       
    data = build_predict_text(text,position_ployphone,sentence_pos,segment)
    result=[]
    with torch.no_grad():
        outputs,_ = model(data)
        print(_.shape,_.squeeze(2).squeeze(0).cpu().numpy())
        for num in outputs:
            result.append(id2ployphone[int(num)])
    
    '''
    result=[]
    for pos,ployphone in position:
        print(pos,ployphone)
        start_time = time.clock()    
        data = build_predict_text(text,pos,ployphone,sentence_pos,segment)
        with torch.no_grad():
            outputs,_ = model(data)
            #num = torch.argmax(outputs)
            for num in outputs:
                result.append(id2ployphone[int(num)])
        end_time = time.clock()
        print('one_sentence_time',end_time-start_time)
    '''
    return result

def predict1(text,sentence_pos,segment,position=None,ployphone=None):
    if position==None and ployphone!=None:
        position = [(i,ployphone) for i,s in enumerate(text) if s==ployphone]
    if position==None and ployphone==None:
        position = [(i,s) for i,s in enumerate(text) if ployphone_ch_id.__contains__(s)]
    for pos,ployphone in position:
        data = build_predict_text(text,pos,ployphone,sentence_pos,segment)
        #print('data shape ',data)
    return data
def predict2(content,sentence_pos,segment,position=None,ployphone=None):
    #print(content,sentence_pos,position)
    data = build_predict_text2(content,position,ployphone,sentence_pos,segment)
    with torch.no_grad():
        outputs,_ = model(data)
        #print('output ',outputs.shape)
        num = torch.argmax(outputs)
    #print(num)    
    return id2ployphone[int(num)]

def file_pred(file_name):
    file_in = open(file_name,'r',encoding='utf-8')
    lines = file_in.readlines()
    true_label_number=0
    error_label_number=0
    file_out = open('error_file.txt','w+',encoding='utf-8')
    file_out.write('{}\t{}\t{}\n'.format('文本','错误标签','正确标签'))
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if 'label' in line:
            continue
        #number,content,position, ployphone,label_ = line.split('\t')
        number,content,position,sentence_pos,segment,ployphone,label_ = line.split('\t')
        sentence_pos = sentence_pos[1:-1].split(',')
        segment = segment[1:-1].split(',')
        sentence_pos = list(map(int, sentence_pos))
        sentence_pos.insert(0,0)
        segment = list(map(int, segment))
        segment.insert(0,0)
        start_time=time.clock()
        pred_label = predict2(content,sentence_pos,segment,position=position,ployphone=ployphone)
        #print(pred_label,label_)
        #print(time.clock()-start_time)
        if len(pred_label)==0:
            print('模型预测的为空值')
        if label_==pred_label:
            true_label_number+=1
        else:
            file_out.write('{}\t{}\t{}\n'.format(content,pred_label,label_))
            error_label_number+=1
    file_out.close()
    print('准确率为 ',true_label_number/(true_label_number+error_label_number))


def input_message():
    ch = ''
    while True:
        chinese_temp = input('（stop all 结束全部；next下一次输入）请输入：')
        if chinese_temp == 'stop all':
            return ''
        if chinese_temp == 'next':
            ch = ch.encode('utf-8', errors='surrogateescape').decode('utf-8')
            return ch
        ch = ch + chinese_temp + '\n'

def input_predict():
    while True:
        chinese = input_message()
        if len(chinese) == 0:
            break
        if '|' in chinese:
            texts,ployphones = chinese.strip().split('|')
        else:
            texts = chinese.strip()
            ployphones = None
        start_time = time.clock()
        segment,sentence_pos = get_chinese_segment(texts)
        results = predict(texts,sentence_pos,segment,ployphone=ployphones)
        print('one sentence cost time is {} s\n'.format(str(time.clock()-start_time)))
        print('the input is ',texts,ployphones)
        print('the output is',results)

if __name__ == '__main__':
   
    input_predict()
    #file_pred('haitian_data/data_100/test.csv') 
