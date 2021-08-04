# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.csv'                                # 训练集
        self.dev_path = dataset + '/data/dev.csv'                                    # 验证集
        self.test_path = dataset + '/data/test.csv'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 30                                             # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size =64 #32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768 #256 #768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768 
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.convs_position =nn.ModuleList([nn.Linear(config.pad_size,config.hidden_size)])
        ###定义FNN
        self.fc1 = nn.Linear(config.pad_size,config.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_size,config.hidden_size)
        ####FNN结束
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)

        #self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def conv_and_pool(self, x, conv):
        #print('conv pool ',x.shape)
        #print('conv ' ,conv(x).shape)
        x = F.relu(conv(x)).squeeze(3)
        #print('relu ',x.shape)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #print('pool ',x.shape)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
       # print(len(x))
        #print(x[3])
        polyphones = x[3]
        positions = x[4]
        #print(positions.shape)
        #print(positions.dtype)
        #print(context.dtype)
        #positions = positions.unsqueeze(1)        
        #positions = positions.unsqueeze(1)
        ###FNNforward
        out_position=self.fc1(positions)
        out_position=self.relu(out_position)
        out_position=self.fc2(out_position)
        #print(out_position.shape)

        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        #print('encodering ',encoder_out.shape)
        #out = encoder_out.unsqueeze(1)
        #print('unsqueeze ',out.shape)
       # for conv in self.convs:
        #    print('convs ',conv ,self.conv_and_pool(out, conv).shape)
        #out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        
        #print('conv',out.shape)
        #out=out.unsqueeze(1)
        out_position = out_position.unsqueeze(1)
        out = torch.cat((encoder_out,out_position),dim=1)
        #print('lstm before',out.shape)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        #print('dropout',out.shape)
        #out = self.fc_cnn(out)
        out = self.fc_rnn(out[:, -1, :])
        polyphones = polyphones.type_as(out)
        #print(out.shape,polyphones.shape)
        out = polyphones*torch.exp(out)/torch.sum(polyphones*torch.exp(out),dim=1).view(-1,1)
        #print('after ',out)
        #print('out.shape ',out.shape)
        return out
