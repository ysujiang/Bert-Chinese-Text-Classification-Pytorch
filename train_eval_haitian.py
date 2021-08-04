# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils_haitian import get_time_dif
from utils_haitian import *
from pytorch_pretrained.optimization import BertAdam
from myloss import ModifiedFocalLoss,MyCrossEntropyLoss
import pandas as pd
from torch.optim import lr_scheduler

MFL=ModifiedFocalLoss()
#MFL = MyCrossEntropyLoss()

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def my_loss(outputs,labels):
    #'''
    log_out = torch.log(outputs)
    log_out = torch.where(torch.isinf(log_out), torch.full_like(log_out, 0), log_out)
    loss_function = (1-outputs)**0.7*log_out
    loss = F.nll_loss(loss_function,labels)
    '''
    log_out = F.log_softmax(outputs, 1)
    loss = F.nll_loss(log_out,labels)
    '''
    return loss
def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    #optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #'''
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    #'''
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[2,3,4,5],gamma = 0.1) #衰减学习率jiang
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        scheduler.step()
        print(epoch, scheduler.get_lr()[0])
        for i, (trains, labels) in enumerate(train_iter):
            outputs,positions = model(trains)
            model.zero_grad()
            #print('output ',outputs.dtype,outputs.shape)
            #print('label ',labels.dtype,labels.shape)
            #loss = F.cross_entropy(outputs, labels)
            '''
            log_out = torch.log(outputs)
            log_out = torch.where(torch.isinf(log_out), torch.full_like(log_out, -100), log_out)
            loss_function = (1-outputs)**0.7*log_out
            print(loss_function)
            loss = F.nll_loss(loss_function,labels)
            print(loss)
            print(loss.shape)
            print(loss.dtype)
            '''
            
            loss = MFL(outputs, labels)
            
            #print(np.any(np.isnan(outputs.numpy)))
            
            
            loss.backward()
            optimizer.step()
            #scheduler.step()#衰减学习率jiang
            
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                aa=msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve)
                if 'nan' in aa:
                    print('nan is here')
                    print(predic)
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                #print(epoch, scheduler.get_lr()[0])
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    if test:
        f_out = open(config.model_name+'.txt','w+',encoding='utf-8')
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs,positions = model(texts)
            #print(outputs)
            #print('text ',texts[0][0])
            token = config.tokenizer.convert_ids_to_tokens(texts[0][0].cpu().numpy())
            #print(token)
            #loss = F.cross_entropy(outputs, labels)
            loss = MFL(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            #'''
            if test:
                #f_out = open(config.model_name+'.txt','w+',encoding='utf-8')
                for i in range(len(texts[0])):
                    #print(i)
                    #print('predic is {} , true label is {} .'.format(id2ployphone[int(predic[i])],id2ployphone[int(labels[i])]))
                    if predic[i]!=labels[i]:
                        #print(predic[i],labels[i])
                        f_out.write('predic is {} , true label is {} ,predic id {},true id {} . \n'.format(id2ployphone[int(predic[i])],id2ployphone[int(labels[i])],predic[i],labels[i]))
                        position_erroe = list(positions[i].cpu().numpy()).index(1.0)
                        f_out.write('position_error {} \n'.format(position_erroe))
                        list_error = config.tokenizer.convert_ids_to_tokens(texts[0][i].cpu().numpy()[:int(texts[1][i])])
                        f_out.write(str(list_error[position_erroe:position_erroe+3])+'\n')
                        f_out.write(str(list_error)+'\n')
                        f_out.write('===========================================\n')
            #'''
    acc = metrics.accuracy_score(labels_all, predict_all)
    #print('acc is ',acc)
    if test:
        #print('acc is ',acc)
        #print(len(set(labels_all)),len(set(predict_all)))
        #print(set(labels_all))
        #print(set(predict_all))
        #print(config.class_list)
        report = metrics.classification_report(labels_all, predict_all)#, target_names=list(set(labels_all)),digits=4)#config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        f_out.write(str(acc)+'\n')
        f_out.write(str(loss_total/len(data_iter))+'\n')
        f_out.write(str(report)+'\n')
        #print(len(report.split('\n'))) #[9].split('       ')[1])
        #print('len ',len(report))
        #df_report = pd.DataFrame(report)
        #print(df_report.head(5))
        report_split = report.split('\n')
        row_name=[]
        for rsp in report_split:
            rs_split = rsp.split('      ')
            #print('join ',' '.join(rs_split))
            #print(len(rs_split))
            if len(rs_split)==6:
                if rs_split[1].replace(' ','').isdigit():
                #print('666 is ',rs_split[1].replace(' ',''))
                    row_name.append(id2ployphone[int(rs_split[1].replace(' ',''))])
                #else:
                    #print('not number',rs_split)
        for aa in confusion:
            #print(aa)
            f_out.write(str(aa).replace('\n','')+'\n')
        #print('row name ',row_name)
        f_out.close()
        df = pd.DataFrame(confusion)
        df.index=row_name
        df.columns=row_name
        df.to_csv(config.model_name+'.csv')
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
