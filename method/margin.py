import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.nn import Parameter

class AMSoftmaxLoss(nn.Module):

    def __init__(self, hidden_dim, speaker_num, s=30.0, m=0.4, **kwargs):
        '''
        AM Softmax Loss
        '''
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.speaker_num = speaker_num
        self.W = torch.nn.Parameter(torch.randn(hidden_dim, speaker_num), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x_BxH, labels_B):
        #print('x_BxH', x_BxH.size())
        #print('labels_B', labels_B.size())
        '''
        x shape: (B, H)
        labels shape: (B)
        '''
        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        W = F.normalize(self.W, dim=0)

        x_BxH = F.normalize(x_BxH, dim=1)

        wf = torch.mm(x_BxH, W)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels_B]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels_B)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, use_label_smoothing=True):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.to(torch.device('cuda'))
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class AM_Softmax_v2(nn.Module): #requires classification layer for normalization 
    def __init__(self, m=0.35, s=30, d=2048, num_classes=625, use_gpu=True, epsilon=0.1):
        super(AM_Softmax_v2, self).__init__()
        self.m = m
        self.s = s 
        self.num_classes = num_classes
        self.CrossEntropy = CrossEntropyLabelSmooth(self.num_classes , use_gpu=use_gpu)

    def forward(self, features, labels, classifier):
        #print('classifier', classifier)
        '''
        x : feature vector : (b x  d) b= batch size d = dimension 
        labels : (b,)
        classifier : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        # x = torch.rand(32,2048)
        # label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,])
        features = nn.functional.normalize(features, p=2, dim=1) # normalize the features
        with torch.no_grad():
            classifier[-1].weight.div_(torch.norm(classifier[-1].weight, dim=1, keepdim=True))

        cos_angle = classifier(features)
        cos_angle = torch.clamp(cos_angle , min = -1 , max = 1 ) 
        b = features.size(0)
        for i in range(b):
            cos_angle[i][labels[i]] = cos_angle[i][labels[i]]  - self.m 
        weighted_cos_angle = self.s * cos_angle
        #print('weighted_cos_angle', weighted_cos_angle, weighted_cos_angle.size())
        log_probs = self.CrossEntropy(weighted_cos_angle, labels, use_label_smoothing=False)
        #print('log_probs', log_probs)
        return weighted_cos_angle, log_probs


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=30.0, m=0.7, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = Parameter(torch.Tensor(out_features * sub, in_features)).cuda()
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        x = x.cuda()
        cosine = F.linear(F.normalize(x), F.normalize(self.weight)).cuda()
        
        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine, _ = torch.max(cosine, dim=2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

class AAMSoftmaxLoss(nn.Module):
    def __init__(self, hidden_dim, speaker_num, s=15, m=0.3, easy_margin=False, **kwargs):
        super(AAMSoftmaxLoss, self).__init__()
        import math

        self.test_normalize = True
        
        self.m = m
        self.s = s
        self.speaker_num = speaker_num
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.FloatTensor(speaker_num, hidden_dim), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x_BxH, labels_B):

        assert len(x_BxH) == len(labels_B)
        assert torch.min(labels_B) >= 0
        assert torch.max(labels_B) < self.speaker_num
        
        # cos(theta)
        cosine = F.linear(F.normalize(x_BxH), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels_B.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        
        return output
        '''
        loss    = self.ce(output, labels_B)
        return loss
        '''