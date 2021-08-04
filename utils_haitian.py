# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import numpy as np
from data_process.ployphone_process import get_ployphones

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
'''
polyphone_id = {'wei2': 0, 'wei4': 1, 'hai2': 2, 'huan2': 3, 'xing2': 4, 'hang2': 5, 'shu3': 6,
                'shu4': 7, 'bei1': 8, 'bei4': 9, 'chong2': 10, 'zhong4': 11, 'chao2': 12, 'zhao1': 13}
id2ployphone={0:'wei2', 1:'wei4', 2:'hai2', 3:'huan2', 4:'xing2',5: 'hang2', 6:'shu3',7:'shu4', 8:'bei1', 9:'bei4', 10:'chong2', 11:'zhong4', 12:'chao2', 13:'zhao1'}

polyphone_ch_id = {'为': [0, 1], '还': [2, 3], '行': [4, 5], '数': [6, 7], '背': [8, 9], '重': [10, 11], '朝': [12, 13]}
'''
ployphone_id,id2ployphone,ployphone_ch_id = get_ployphones('./data_process/dict_data.txt')
f_out = open('./test.txt','a+',encoding='utf-8')
f_out.write(str(id2ployphone)+'\n')
f_out.close()

def build_dataset(config):

    def load_dataset(path, pad_size=32):
        contents = []
        i=0
        mask_windows=config.windows
        #print(len(ployphone_id))
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                '''
                i+=1
                if i==200:
                    break
                #'''
                if not lin:
                    continue
                if 'label' in lin:
                    continue
                number,content,position,sentence_pos,segment,ployphone,label_ = lin.split('\t')
                #print('len labels',len(label_))
                sentence_pos = sentence_pos[1:-1].split(',')
                segment = segment[1:-1].split(',')
                sentence_pos = list(map(int, sentence_pos))
                sentence_pos.insert(0,0)
                segment = list(map(int, segment))
                segment.insert(0,0)
                #print('pos ',type(sentence_pos),sentence_pos)
                #print('segment ',type(segment),segment)
                label = ployphone_id[label_]
        
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if len(content)>pad_size-1:
                    print(11111111111111)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                        sentence_pos = sentence_pos+[0]*(pad_size-len(sentence_pos)) #词性
                        segment = segment+[0]*(pad_size-len(segment))#分词
                    else:
                        
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                if mask_windows:
                    mask = [0]*pad_size #jiang
               # mask[int(position)-3]=1
                    if len(token)<mask_windows:
                        for index_mask in range(0,len(token)):
                            mask[index_mask]=1
                    elif int(position)-mask_windows//2<0:
                        for index_mask in range(0,mask_windows):
                            mask[index_mask]=1
                    elif int(position)+mask_windows//2+1>len(token):
                        for index_mask in range(1,mask_windows+1):
                            mask[len(token)-index_mask]=1
                    else:
                        for index_mask in range(1,mask_windows//2+1):
                            mask[int(position)-1+index_mask]=1
                            mask[int(position)-1-index_mask]=1
                        mask[int(position)]=1
                #print(mask)
                ploymasked = [0]*len(ployphone_id) #jiang
                for m in ployphone_ch_id[ployphone]: #jiang
                    #print(m,len(ploymasked))
                    #print(ployphone)
                    ploymasked[ployphone_id[m]]=1#jiang
                '''
                positions = [0.0]*len(token_ids)#jiang
                if int(position)>len(token_ids):
                    print(len(content),len(token_ids),position)
                positions[int(position)+1]=1.0#jiang 多了一个开始的位置，因此位置加1
                '''
                positions = [int(position)+1]
                contents.append((token_ids, int(label), seq_len, mask,ploymasked,positions,sentence_pos,segment))
        #print(len(contents[0]))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)#token_id
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)#标签

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)#序列的长度
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)#标记注意力
        ployphones = torch.LongTensor([_[4] for _ in datas]).to(self.device)#jiang多音字的音
        positions = torch.LongTensor([_[5] for _ in datas]).to(self.device)#多音字的位置
        sentence_pos = torch.LongTensor([_[6] for _ in datas]).to(self.device)#句子的词性
        segment = torch.LongTensor([_[7] for _ in datas]).to(self.device)#句子的分词
        return (x, seq_len, mask,ployphones,positions,sentence_pos,segment), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
