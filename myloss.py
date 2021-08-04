import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    def __int__(self, gamma=0.7, beta=0):
        super(ModifiedFocalLoss, self).__init__()
        self.gamma = 0.7
        self.beta = 0

    def forward(self, outputs, labels):
        outputs = torch.clamp(outputs, min=1e-5)
        log_out = torch.log(outputs)
        log_out = torch.where(torch.isinf(log_out), torch.full_like(log_out, 0), log_out)
        loss_function = (1 + 0.5 - outputs) ** 0.7 * log_out
        #print('loss_function ',loss_function)
        loss = F.nll_loss(loss_function, labels)
        #print('loss ',loss)
        return loss


class MyCrossEntropyLoss(nn.Module):
    def __int__(self):
        super(MyCrossEntropyLoss, self).__init__()
        self.alpha = 0.25
        self.gamma = 2

    def forward(self, outputs, labels):
        epsilon = 1e-10
        outputs,labels = self.trans_labels(outputs,labels)
        float_labels = labels.float().view(-1, 1)
        #print(float_labels)
        # outputs0 =outputs
        # outputs = torch.clamp(outputs0, min=1e-5)
        # print(float_labels.shape)
        # print((float_labels*torch.log(outputs+epsilon)).shape)
        # print(((1 - float_labels) * torch.log(1 - outputs + epsilon)).shape)
        # '''
        #cross_entropy_loss = float_labels * torch.log(outputs) + (1 - float_labels) * torch.log(
         #   1 - outputs)
        cross_entropy_loss = float_labels * (1-outputs)**2*torch.log(outputs) + (1 - float_labels) * outputs**2*torch.log(1 - outputs)
        cross_entropy_loss = -cross_entropy_loss
        # print(torch.sum(cross_entropy_loss,1))
        return torch.mean(torch.sum(cross_entropy_loss, 1))
        '''
        output1 = torch.log(outputs)
        loss = F.nll_loss(output1, labels)
        loss2 = torch.mean(-torch.sum((1 - float_labels) * torch.log(1 - outputs0),1))
        return (loss+loss2)/2
        '''

    def trans_labels(self, outputs, labels):
        for i in range(len(outputs)):
            one_sample = (outputs[i] != 0).nonzero()  # 找出非零的概率索引
            condition_label = one_sample.view(1, -1).squeeze(0)  # 找出非零概率所在的索引，进行重新排序
            temp_label = (condition_label == labels[i]).nonzero().squeeze(0)  # 新的标签
            temp_sample = torch.cat([(outputs[i][n]) for n in one_sample]).unsqueeze(0)  # 模型得出的非零概率
            if i ==0:
                new_outputs = temp_sample
                new_labes = temp_label
            else:
                #print('new',new_outputs.shape)
                #print('temp',temp_sample.shape)
                new_outputs = torch.cat((new_outputs, temp_sample), dim=0)
                new_labes = torch.cat((new_labes,temp_label))
        return new_outputs,new_labes
