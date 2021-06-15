from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal
import torch.nn.init as init
import numpy as np
class Encoder_a(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout=0.5):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(Encoder_a, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size*5)
        self.linear_2 = nn.Linear(hidden_size*5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = self.drop(F.relu(self.linear_1(dropped)))
        y_2 = self.drop(F.relu(self.linear_2(y_1)))
        y_3 = F.tanh(self.linear_3(y_2))

        return y_3




class Encoder_5(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout=0.5):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(Encoder_5, self).__init__()
        self.norm = nn.BatchNorm1d(hidden_size)
        self.norm2 = nn.BatchNorm1d(in_size*10)
        self.norm3 = nn.BatchNorm1d(hidden_size*10)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size*10)
        self.linear_2 = nn.Linear(in_size*10, hidden_size*10)
        self.linear_3 = nn.Linear(hidden_size*10, hidden_size)
        self.linear_4 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed =x #self.norm(x)
     #   dropped = self.drop(normed)
        y_1 = F.leaky_relu(self.norm2(self.drop(self.linear_1(normed))))
        y_2 = F.leaky_relu(self.norm3(self.drop(self.linear_2(y_1))))
        y_2 = F.leaky_relu(self.norm(self.drop(self.linear_3(y_2))))
        y_3 = F.tanh(self.linear_4(y_2))

        return y_3



class Encoder_v(nn.Module):

    def __init__(self, in_size, hidden_size, dropout=0.5):

        super(Encoder_v, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size*5)
        self.linear_2 = nn.Linear(hidden_size*5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = self.drop(F.relu(self.linear_1(dropped)))
        y_2 = self.drop(F.relu(self.linear_2(y_1)))
        y_3 = F.tanh(self.linear_3(y_2))

        return y_3



class Encoder_l3(nn.Module):

    def __init__(self, in_size, hidden_size, dropout=0.5):

        super(Encoder_l3, self).__init__()
        self.norm = nn.BatchNorm1d(in_size*5)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size*5, hidden_size*5)
        self.linear_2 = nn.Linear(hidden_size*5, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        KERNEL_SIZE = 5
        self.Gates = nn.Conv1d(1, 5 , KERNEL_SIZE, stride = 1, padding=(KERNEL_SIZE-1)/2)
    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        x = x.unsqueeze(1)
        x = self.Gates(x)
        x = x.view(x.shape[0],-1)
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = self.drop(F.relu(self.linear_1(dropped)))
        y_2 = self.drop(F.relu(self.linear_2(y_1)))
        y_3 = F.tanh(self.linear_3(y_2))
        return y_3


class Encoder_l(nn.Module):
    '''
    The LSTM-based subnetwork that is used in LMF for text
    '''

    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        super(Encoder_l, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze())
        y_1 = F.tanh(self.linear_1(h))
        return y_1




class Decoder2(nn.Module):
    def __init__(self, in_size, out_size):
        super(Decoder2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 64),
            nn.Dropout(0.5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, out_size),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], -1)
        return img




class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        #    nn.Tanh(),
         #   nn.ReLU()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

class classifier2(nn.Module):

    def __init__(self, in_size, output_dim, dropout=0.5):

        super(classifier2, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, output_dim*10)
        self.linear_2 = nn.Linear(output_dim*10, output_dim)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.softmax(self.linear_2(y_1),1)
     #   y_3 = F.tanh(self.linear_3(y_2))

        return y_2


class classifier3(nn.Module):

    def __init__(self, in_size, output_dim, dropout=0.5):

        super(classifier3, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, in_size)
        self.linear_2 = nn.Linear(in_size, output_dim)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = self.drop(F.tanh(self.linear_1(dropped)))
        y_2 = F.softmax(self.linear_2(y_1),1)
     #   y_3 = F.tanh(self.linear_3(y_2))

        return y_2




class graph11_new(nn.Module):

    def __init__(self, in_size, output_dim, hidden = 50, dropout=0.5):

        super(graph11_new, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size*3)
        self.drop = nn.Dropout(p=dropout)
      #  self.graph = nn.Linear(in_size*2, in_size)

        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )


        self.graph_fusion2 = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size*3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
    #    self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
      #  self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

      #  self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
     #   xavier_normal(self.u1)

    def forward(self, x):
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
        ###################### unimodal layer  ##########################
        sa = F.sigmoid(self.attention(a1))
        sv = F.sigmoid(self.attention(v1))
        sl = F.sigmoid(self.attention(l1))

        total_weights = torch.cat([sa, sv, sl],1)
      #  total_weights = torch.cat([total_weights,sl],1)

        unimodal_a = (sa.expand(a1.size(0),self.in_size))
        unimodal_v = (sv.expand(a1.size(0),self.in_size))
        unimodal_l = (sl.expand(a1.size(0),self.in_size))
        sa = sa.squeeze()
        sl = sl.squeeze()
        sv = sv.squeeze()
        unimodal = (unimodal_a * a1 + unimodal_v * v1 + unimodal_l * l1)/3

        ##################### bimodal layer ############################
        a = F.softmax(a1, 1)
        v = F.softmax(v1, 1)
        l = F.softmax(l1, 1)
        sav = (1/(torch.matmul(a.unsqueeze(1), v.unsqueeze(2)).squeeze() +0.5) *(sa+sv))
        sal = (1/(torch.matmul(a.unsqueeze(1), l.unsqueeze(2)).squeeze() +0.5) *(sa+sl))
        svl = (1/(torch.matmul(v.unsqueeze(1), l.unsqueeze(2)).squeeze() +0.5) *(sl+sv))
     #   print('sav',sav.shape)
        normalize = torch.cat([sav.unsqueeze(1), sal.unsqueeze(1), svl.unsqueeze(1)],1)
        normalize = F.softmax(normalize,1)
        total_weights = torch.cat([total_weights,normalize],1)
    #    print('normalize',normalize.shape)
     #   print((normalize[:,0].unsqueeze(1).expand(a.size(0),self.in_size)).shape,'shape')
        a_v = F.elu((normalize[:,0].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([a1,v1],1)))
        a_l = F.elu((normalize[:,1].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([a1,l1],1)))
        v_l = F.elu((normalize[:,2].unsqueeze(1).expand(a.size(0), self.in_size)) * self.graph_fusion(torch.cat([v1,l1],1)))
        bimodal = (a_v + a_l + v_l)
    
        ###################### trimodal layer ####################################
        a_v2 = F.softmax(self.graph_fusion(torch.cat([a1,v1],1)), 1)
        a_l2 = F.softmax(self.graph_fusion(torch.cat([a1,l1],1)), 1)
        v_l2 = F.softmax(self.graph_fusion(torch.cat([v1,l1],1)), 1)
        savvl = (1/(torch.matmul(a_v2.unsqueeze(1), v_l2.unsqueeze(2)).squeeze() +0.5) *(sav+svl))
        saavl = (1/(torch.matmul(a_v2.unsqueeze(1), a_l2.unsqueeze(2)).squeeze() +0.5) *(sav+sal))
        savll = (1/(torch.matmul(a_l2.unsqueeze(1), v_l2.unsqueeze(2)).squeeze() +0.5) *(sal+svl))
        savl = (1/(torch.matmul(a_v2.unsqueeze(1), l.unsqueeze(2)).squeeze() +0.5) *(sav+sl))
        salv = (1/(torch.matmul(a_l2.unsqueeze(1), v.unsqueeze(2)).squeeze() +0.5) *(sal+sv))
        svla = (1/(torch.matmul(v_l2.unsqueeze(1), a.unsqueeze(2)).squeeze() +0.5) *(sa+svl))

        normalize2 = torch.cat([savvl.unsqueeze(1), saavl.unsqueeze(1), savll.unsqueeze(1), savl.unsqueeze(1), salv.unsqueeze(1), svla.unsqueeze(1)],1)
        normalize2 = F.softmax(normalize2,1)
        total_weights = torch.cat([total_weights,normalize2],1)
       # print((normalize2[:,0].unsqueeze(1).expand(a.size(0),self.in_size)).shape,'shape')
        avvl = F.elu((normalize2[:,0].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_v,v_l],1)))
        aavl = F.elu((normalize2[:,1].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_v,a_l],1)))
        avll = F.elu((normalize2[:,2].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([v_l,a_l],1)))
        avl = F.elu((normalize2[:,3].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_v,l1],1)))
        alv = F.elu((normalize2[:,4].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([a_l,v1],1)))
        vla = F.elu((normalize2[:,5].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion2(torch.cat([v_l,a1],1)))
        trimodal = (avvl + aavl + avll + avl + alv + vla)
        fusion = torch.cat([unimodal,bimodal],1)
        fusion = torch.cat([fusion,trimodal],1)        
        fusion = self.norm2(fusion)
     #   fusion = self.drop(fusion)
        y_1 = F.tanh(self.linear_1(fusion))
        y_1 = F.tanh(self.linear_2(y_1))
        y_2 = F.softmax(self.linear_3(y_1),1)
     #   y_3 = F.tanh(self.linear_3(y_2))

        return y_2, total_weights





class concat(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(concat, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size*3)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)

        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )


        self.graph_fusion2 = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size*3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
        xavier_normal(self.u1)

    def forward(self, x):
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
     
        fusion = torch.cat([a1,v1],1)
        fusion = torch.cat([fusion,l1],1)        
        fusion = self.norm2(fusion)
     #   fusion = self.drop(fusion)
        y_1 = F.tanh(self.linear_1(fusion))
        y_1 = F.tanh(self.linear_2(y_1))
        y_2 = F.softmax(self.linear_3(y_1),1)
     #   y_3 = F.tanh(self.linear_3(y_2))

        return y_2, y_2



class multiplication(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(multiplication, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
        xavier_normal(self.u1)

    def forward(self, x):
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
     
        fusion = a1*v1
        fusion = v1*l1        
        fusion = self.norm2(fusion)
     #   fusion = self.drop(fusion)
        y_1 = F.tanh(self.linear_1(fusion))
        y_1 = F.tanh(self.linear_2(y_1))
        y_2 = F.softmax(self.linear_3(y_1),1)
     #   y_3 = F.tanh(self.linear_3(y_2))

        return y_2, y_2


class tensor(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(tensor, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, 1)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
        xavier_normal(self.u1)

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        DTYPE = torch.cuda.FloatTensor
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
           
        _audio_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), a1), dim=1)
        _video_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), v1), dim=1)
        _text_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), l1), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(-1, (a1.size(1) + 1) * (a1.size(1) + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(a1.size(0), -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.softmax(self.post_fusion_layer_3(post_fusion_y_2))
      #  output = post_fusion_y_3 * self.output_range + self.output_shift
        y_2 = post_fusion_y_3


        return y_2, y_2


class low_rank(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(low_rank, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, 1)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.rank = 4
        self.output_dim = output_dim
        self.audio_factor = Parameter(torch.Tensor(self.rank, in_size + 1, output_dim).cuda())
        self.video_factor = Parameter(torch.Tensor(self.rank, in_size + 1, output_dim).cuda())
        self.text_factor = Parameter(torch.Tensor(self.rank, in_size + 1, output_dim).cuda())
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank).cuda())
        self.fusion_bias = Parameter(torch.Tensor(1, output_dim).cuda())


        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear((in_size + 1) * (in_size + 1) * (in_size + 1), hidden)
        self.post_fusion_layer_2 = nn.Linear(hidden, hidden)
        self.post_fusion_layer_3 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
        DTYPE = torch.cuda.FloatTensor
        _audio_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), a1), dim=1)
        _video_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), v1), dim=1)
        _text_h = torch.cat((Variable(torch.ones(a1.size(0), 1).type(DTYPE), requires_grad=False), l1), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_video * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        y_2 = output.view(-1, self.output_dim)


        return y_2, y_2



class late(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(late, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)


        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

        self.u1 = Parameter(torch.Tensor(in_size, in_size).cuda())
        xavier_normal(self.u1)

    def forward(self, x):
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
           
        a1 = self.norm2(a1)
        a = F.tanh(self.attention(a1))

        v1 = self.norm2(v1)
        v = F.tanh(self.attention(v1))

        l1 = self.norm2(l1)
        l = F.tanh(self.attention(l1))

        fusion = torch.cat([a,v],1)
        fusion = F.softmax(torch.cat([fusion,l],1))

        fusion = fusion[:,0].unsqueeze(1).expand(a1.size(0),self.in_size) * a1 + fusion[:,1].unsqueeze(1).expand(a1.size(0),self.in_size) * v1 + fusion[:,2].unsqueeze(1).expand(a1.size(0),self.in_size) * l1 

        fusion = self.norm2(fusion)
     #   fusion = self.drop(fusion)
        y_1 = F.tanh(self.linear_1(fusion))
        y_1 = F.tanh(self.linear_2(y_1))
        y_2 = F.softmax(self.linear_3(y_1),1)

        return y_2, y_2



class graph12(nn.Module):

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5):

        super(graph12, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.norm2 = nn.BatchNorm1d(in_size*3)
        self.drop = nn.Dropout(p=dropout)
        self.graph = nn.Linear(in_size*2, in_size)

        self.graph_fusion = nn.Sequential(
            nn.Linear(in_size*2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, in_size),
            nn.Tanh()
        )

        self.attention = nn.Linear(in_size, 1)
        self.linear_1 = nn.Linear(in_size*3, hidden)
        self.linear_2 = nn.Linear(hidden, hidden)
        self.linear_3 = nn.Linear(hidden, output_dim)
        self.rnn = nn.LSTM(in_size, hidden, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)
  #      self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.lstm1 = nn.LSTMCell(in_size,hidden)
        self.hidden = hidden
        self.in_size = in_size

    def forward(self, x):
        a1 = x[:,0,:]; v1 = x[:,1,:]; l1 = x[:,2,:]
        sa = F.tanh(self.attention(a1))
        sv = F.tanh(self.attention(v1))
        sl = F.tanh(self.attention(l1))

        normalize = torch.cat([sa,sv],1)
        normalize = torch.cat([normalize,sl],1)
        normalize = F.softmax(normalize,1)
        sa = normalize[:,0].unsqueeze(1)
        sv = normalize[:,1].unsqueeze(1)
        sl = normalize[:,2].unsqueeze(1)

        total_weights = torch.cat([sa,sv],1)
        total_weights = torch.cat([total_weights,sl],1)

        unimodal_a = (sa.expand(a1.size(0),self.in_size))
        unimodal_v = (sv.expand(a1.size(0),self.in_size))
        unimodal_l = (sl.expand(a1.size(0),self.in_size))
        sa = sa.squeeze()
        sl = sl.squeeze()
        sv = sv.squeeze()
        unimodal = (unimodal_a * a1 + unimodal_v * v1 + unimodal_l * l1)/3
        a = F.softmax(x[:,0,:],1).unsqueeze(1)
        v = F.softmax(x[:,1,:],1).unsqueeze(2)
        l = F.softmax(x[:,2,:],1).unsqueeze(2)
        sav = (1/(torch.matmul(a,v).squeeze()+0.5)*(sa+sv))
        sal = (1/(torch.matmul(a,l).squeeze()+0.5)*(sa+sl))
        svl = (1/(torch.matmul(v.squeeze().unsqueeze(1),l).squeeze()+0.5)*(sl+sv))
     #   print('sav',sav.shape)
        normalize = torch.cat([sav.unsqueeze(1),sal.unsqueeze(1)],1)
        normalize = torch.cat([normalize,svl.unsqueeze(1)],1)
        normalize = F.softmax(normalize,1)
        total_weights = torch.cat([total_weights,normalize],1)
    #    print('normalize',normalize.shape)
     #   print((normalize[:,0].unsqueeze(1).expand(a.size(0),self.in_size)).shape,'shape')
        a_v = F.leaky_relu((normalize[:,0].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a1,v1],1)))
        a_l = F.leaky_relu((normalize[:,1].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a1,l1],1)))
        v_l = F.leaky_relu((normalize[:,2].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([v1,l1],1)))
        bimodal = (a_v + a_l + v_l)/3
        a_v2 = F.softmax(a_v,1).unsqueeze(1)
        a_l2 = F.softmax(a_l,1).unsqueeze(2)
        v_l2 = F.softmax(v_l,1).unsqueeze(2)
        savvl = (1/(torch.matmul(a_v2,v_l2).squeeze()+0.5)*(sav+svl))
        saavl = (1/(torch.matmul(a_v2,a_l2).squeeze()+0.5)*(sav+sal))
        savll = (1/(torch.matmul(a_l2.squeeze().unsqueeze(1),v_l2).squeeze()+0.5)*(sal+svl))
        savl = (1/(torch.matmul(a_v2,l).squeeze()+0.5)*(sav+sl))
        salv = (1/(torch.matmul(a_l2.squeeze().unsqueeze(1),v).squeeze()+0.5)*(sal+sv))
        svla = (1/(torch.matmul(v_l2.squeeze().unsqueeze(1),a.squeeze().unsqueeze(2)).squeeze()+0.5)*(sa+svl))

        normalize2 = torch.cat([savvl.unsqueeze(1),saavl.unsqueeze(1)],1)
        normalize2 = torch.cat([normalize2,savll.unsqueeze(1)],1)
        normalize2 = torch.cat([normalize2,savl.unsqueeze(1)],1)
        normalize2 = torch.cat([normalize2,salv.unsqueeze(1)],1)
        normalize2 = torch.cat([normalize2,svla.unsqueeze(1)],1)
        normalize2 = F.softmax(normalize2,1)
        total_weights = torch.cat([total_weights,normalize2],1)
       # print((normalize2[:,0].unsqueeze(1).expand(a.size(0),self.in_size)).shape,'shape')
        avvl = F.leaky_relu((normalize2[:,0].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a_v,v_l],1)))
        aavl = F.leaky_relu((normalize2[:,1].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a_v,a_l],1)))
        avll = F.leaky_relu((normalize2[:,2].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([v_l,a_l],1)))
        avl = F.leaky_relu((normalize2[:,3].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a_v,l1],1)))
        alv = F.leaky_relu((normalize2[:,4].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([a_l,v1],1)))
        vla = F.leaky_relu((normalize2[:,5].unsqueeze(1).expand(a.size(0),self.in_size)) * self.graph_fusion(torch.cat([v_l,a1],1)))
        trimodal = (avvl + aavl + avll + avl + alv + vla)/6
        fusion = torch.cat([unimodal,bimodal],1)
        fusion = torch.cat([fusion,trimodal],1)        
        fusion = self.norm2(fusion)
        fusion = self.drop(fusion)
        y_1 = F.tanh(self.linear_1(fusion))
        y_1 = F.tanh(self.linear_2(y_1))
        y_2 = F.softmax(self.linear_3(y_1),1)
     #   y_3 = F.tanh(self.linear_3(y_2))

        return y_2, total_weights


class outer_product(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, in_size, output_dim,hidden = 50, dropout=0.5, use_softmax=True):

        super(outer_product, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = in_size
        self.video_in = in_size
        self.text_in = in_size

        self.audio_hidden = hidden
        self.video_hidden = hidden
        self.text_hidden = hidden
        self.output_dim = output_dim
        self.use_softmax = use_softmax

        self.audio_prob = dropout
        self.video_prob = dropout
        self.text_prob = dropout
        self.post_fusion_prob = dropout

        # define the pre-fusion subnetworks

        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.audio_in + 1) * (self.video_in + 1) * (self.text_in + 1), self.audio_hidden)
        self.post_fusion_layer_2 = nn.Linear(self.audio_hidden, self.audio_hidden)
        self.post_fusion_layer_3 = nn.Linear(self.audio_hidden, output_dim)


    def forward(self, x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = x[:,0,:]; video_h = x[:,1,:]; text_h = x[:,2,:]
        batch_size = audio_h.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_in + 1) * (self.video_in + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output, x

