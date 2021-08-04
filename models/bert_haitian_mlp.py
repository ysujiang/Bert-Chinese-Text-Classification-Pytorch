# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert_mlp_haitian_67_3'
        self.train_path = dataset + '/train.csv'                                # 训练集
        self.dev_path = dataset + '/dev.csv'                                    # 验证集
        self.test_path = dataset + '/test.csv'                                  # 测试集
        self.class_list = [x.strip() for x in open('./data_process/class.txt').readlines()]
            #dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        #self.device = torch.device('cpu')
        self.windows = 3 #False
        self.require_improvement = 5000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数，进行去重
        self.num_epochs = 30 #10 #30#30                                             # epoch数
        self.batch_size = 128 #64 #128 #32 #64                                           # mini-batch大小
        self.pad_size = 51 #9000 #64 #32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.pos_embedding = nn.Embedding(num_embeddings=59, embedding_dim = config.hidden_size)#58个词性
        self.segment_embedding = nn.Embedding(num_embeddings=4, embedding_dim = config.hidden_size)#4个分词
        self.position_embedding = nn.Embedding(num_embeddings=3,embedding_dim= config.hidden_size)#位置
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size*4)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        # 定义MFNN网络
        self.net_mfnn = nn.Sequential()
        self.net_mfnn.add_module('layer_1', nn.Linear(config.hidden_size*4, 512))
        self.net_mfnn.add_module('layer_2', nn.ReLU())
        self.net_mfnn.add_module('layer_3', nn.Linear(512, 1024))
        self.net_mfnn.add_module('layer_4', nn.ReLU())
        self.net_mfnn.add_module('layer_7', nn.Linear(1024, 512))
        self.net_mfnn.add_module('layer_4',nn.Tanh())
        self.mfnn_fc = nn.Linear(512*config.pad_size,config.num_classes)
        #self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        polyphones = x[3]
        positions = x[4]
        sentence_pos=x[5]
        segment = x[6]
    
        #embedding
        pos_embedding = self.pos_embedding(sentence_pos)
        segment_embedding = self.segment_embedding(segment)
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        positions = self.position_embedding(positions)
        encoder_out = torch.cat((encoder_out,positions,pos_embedding,segment_embedding),dim=2)#jiang
        #分类器
        #print('shape encoder,',encoder_out.shape)
        out = self.net_mfnn(encoder_out)
        out = self.dropout(out)
        #print(out.shape)
        out = out.view(out.size(0),-1)
        out = self.mfnn_fc(out)
        #print(out.shape)
        '''
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        '''
        polyphones = polyphones.type_as(out)
        #print('sum shape ',torch.sum(polyphones*torch.exp(out),dim=1).shape)
        out = polyphones*torch.exp(out)/torch.sum(polyphones*torch.exp(out),dim=1).view(-1,1)
        return out,x[4]
