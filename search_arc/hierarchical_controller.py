# hierarchical_controller.py
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTM_index(torch.nn.Module):
    """
    in paper is LSTM_1  generate  index connect 

    """
    def __init__(self,blocks):
        torch.nn.Module.__init__(self)
        #self.args = args
        # TODO
        self.blocks = blocks
        self.num_branches = 5 # operation total type 
        self.num_cells = 5 #node num
        self.lstm_size =256
        self.lstm_num_layers = 2
        self.temperature = 5 # let probability smooth 
        self.tanh_constant = 1.1 #let probability smooth 
        self.op_tanh_reduce = 2.5 #et probability smooth 

        self.encoder = nn.Embedding(self.num_branches, self.lstm_size)

        self.lstm = nn.LSTMCell(self.lstm_size, self.lstm_size)
        # attention
        #w_attn_2 = paper mlp f1 , v_attn = paper mlp f2 ,  w_attn_1 attention not in paper 
        self.w_attn_1 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.w_attn_2 = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        self.all_prob = []
        #init_values
    def forward(self):
        c, h = None , None
        all_arc = []
        all_entropy = 0
        all_log_prob = 0
        for i in range(self.blocks):
            arc_seq, entropy, log_prob, c, h = self.run_sampler(prev_c=c,prev_h=h)
            all_arc.append(arc_seq)
            all_entropy = all_entropy + entropy
            all_log_prob = all_log_prob + log_prob
        return all_arc, all_log_prob, all_entropy , c , h

    def run_sampler(self, prev_c=None, prev_h=None):
        if (prev_c is not None) & (prev_h is not None):
            # TODO: multi-layer LSTM
            #prev_c = [torch.zeros(1, self.lstm_size).cuda() for _ in range(self.lstm_num_layers)]
            #prev_h = [torch.zeros(1, self.lstm_size).cuda() for _ in range(self.lstm_num_layers)]
            self.prev_c =  torch.zeros(1, self.lstm_size).cuda()
            self.prev_h =  torch.zeros(1, self.lstm_size).cuda()

        inputs = self.encoder(torch.zeros(1,dtype=torch.long).cuda())

        anchors = []
        anchors_w_1 = []
        arc_seq = []
        all_prob = []
        for layer_id in range(2):
            embed = inputs
            next_h, next_c = self.lstm(embed, ( torch.zeros(1, self.lstm_size).cuda(),  torch.zeros(1, self.lstm_size).cuda()))
            prev_c, prev_h = next_c, next_h
            anchors.append(torch.zeros(1,self.lstm_size).cuda())
            anchors_w_1.append(self.w_attn_1(next_h))

        layer_id = 2
        entropy = []
        log_prob = []

        while layer_id < self.num_cells + 2:
            prev_layers = []
            for i in range(2): # index_1, index_2
                embed = inputs
                next_h, next_c = self.lstm(embed, (prev_h, prev_c))
                prev_c, prev_h = next_c, next_h
                #
                query = torch.stack(anchors_w_1[:layer_id], dim=1)
                query = query.view(layer_id, self.lstm_size)

                query = torch.tanh(query + self.w_attn_2(next_h))
                query = self.v_attn(query)
                logits = query.view(1, layer_id)
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    logits = self.tanh_constant * torch.tanh(logits)
                prob = F.softmax(logits, dim=-1)
                all_prob.append(prob.cpu().detach().numpy())
                #print(prob)
                index = torch.multinomial(prob, 1).long().view(1)
                arc_seq.append(index)
                #arc_seq.append(0)
                curr_log_prob = F.cross_entropy(logits, index)
                log_prob.append(curr_log_prob)
                curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(logits, dim=-1), prob), dim=1)).detach()

                entropy.append(curr_ent)
                prev_layers.append(anchors[index])
                inputs = prev_layers[-1].view(1, -1).requires_grad_()

            next_h, next_c = self.lstm(inputs, (prev_h, prev_c))
            prev_c, prev_h = next_c, next_h
            anchors.append(next_h)
            anchors_w_1.append(self.w_attn_1(next_h))

            inputs = self.encoder(torch.zeros(1,dtype=torch.long).cuda())
            layer_id += 1

        arc_seq = torch.tensor(arc_seq,dtype=torch.long)
        entropy = sum(entropy)
        log_prob = sum(log_prob)
        last_c = next_c
        last_h = next_h
        self.all_prob = all_prob
        return arc_seq, entropy, log_prob, last_c, last_h
class LSTM_operation(torch.nn.Module):
    """
    in paper is LSTM_2  generate operation 

    """
    def __init__(self,blocks):
        torch.nn.Module.__init__(self)
        #self.args = args
        # TODO
        self.blocks = blocks
        self.num_branches = 5
        self.num_cells = 5
        self.lstm_size =256
        self.lstm_num_layers = 1
        self.temperature = 5

        self.tanh_constant = 1.1
        self.op_tanh_reduce = 2.5
        self.op_tanh = self.tanh_constant / self.op_tanh_reduce
        #encode input index
        self.encoder = nn.Embedding(self.num_branches+1, self.lstm_size)
        #master architecture
        self.lstm_oper = nn.LSTM(self.lstm_size, self.lstm_size,num_layers = self.lstm_num_layers,bidirectional=True)

        self.fc = nn.Linear(self.lstm_size *2,self.num_branches)
        self.softmax = nn.Softmax(dim=1)
        self.all_prob = []

        # self.register_buffer('h0', torch.randn(2,1,64)/100)
        # self.register_buffer('c0', torch.randn(2,1,64)/100)
        # self.h0 = (torch.randn(2,1,64)/100).cuda()
        # self.c0 = (torch.randn(2,1,64)/100).cuda()

    def forward(self,arc,c,h):
        c, h = None , None
        all_arc = []
        all_entropy = 0
        all_log_prob = 0
        for i in range(self.blocks):
            arc_seq, entropy, log_prob, c, h = self.run_sampler(arc[i],c=c,h=h)
            all_entropy = all_entropy + entropy
            all_log_prob = all_log_prob + log_prob
            all_arc.append(arc_seq)
        return all_arc, all_log_prob, all_entropy , c , h

    def run_sampler(self,arc,c=None,h=None):
        entropy = []
        log_prob = []
        if (c is  None ):
            #self.h0,self.c0 = c , h
            c = (torch.randn(2,1,self.lstm_size)/100).cuda()
            h = (torch.randn(2,1,self.lstm_size)/100).cuda()

        arc = self.encoder(arc).unsqueeze(1)
        #C_n, H_n if need create varible
        arc,(h,c) = self.lstm_oper(arc,(c,h))
        arc = self.fc(arc)
        if self.temperature is not None:
            arc /= self.temperature
        if self.tanh_constant is not None:
            arc = self.op_tanh * torch.tanh(arc)
        prob = F.softmax(arc, dim=-1)
        self.all_prob.append(prob.cpu().detach().numpy())
        index = torch.multinomial(prob.squeeze(), 1).long().view(-1)
        curr_log_prob = F.cross_entropy(arc.squeeze(), index.squeeze())
        log_prob.append(curr_log_prob)
        curr_ent = -torch.mean(torch.sum(torch.mul(F.log_softmax(arc, dim=-1), prob), dim=1)).detach()
        entropy.append(curr_ent)
        entropy = sum(entropy)
        log_prob = sum(log_prob)
        return index, entropy, log_prob, c, h

class separable_LSTM(torch.nn.Module):
    def __init__(self,blocks):
        torch.nn.Module.__init__(self)
        self.blocks = blocks
        self.device = 'cuda'
        self.index_genreate = LSTM_index(blocks).to(self.device)
        self.node_genertate = LSTM_operation(blocks).to(self.device)
    def forward(self):
        dag_ind, log_prob_ind, entropy_ind , c , h  = self.index_genreate()
        #print(dag_ind)
        dag_ind = [i.to(self.device) for i in dag_ind]
        dag_node, log_prob_node, entropy_node , c , h=self.node_genertate(dag_ind,c,h)
        arc = []
        for i in range(self.blocks):
            arc.append(self.index_operation_merge(dag_ind[i].cpu(),dag_node[i].cpu()))
        return arc , (log_prob_ind , log_prob_node) , (entropy_ind , entropy_node)
    def index_operation_merge(self,dag_ind,dag_node):
        device = ('cpu' if dag_ind[0].get_device() == -1 else 'cuda')
        return torch.cat((dag_ind.to(device).unsqueeze(1),dag_node.to(device).unsqueeze(1)),dim=1).flatten().tolist()
    def prob_get(self):
        return self.index_genreate.all_prob , self.node_genertate.all_prob

# from  tqdm import tqdm
# for i in tqdm(range(1)):
#     test = separable_LSTM(8)
#     dag, log_prob, entropy= test()
#     print(dag)
    #print(test.prob_get())
    #model = model_maker(dag)
# test = separable_LSTM()
# test.index_genreate.train()
# test.node_genertate.train()
# print(test.index_genreate.training,test.node_genertate.training)

# controller_optimizer = torch.optim.Adam(
#         test.parameters(),
#         0.01,
#         betas=(0.1,0.999),
#         eps=1e-3,
#     )
# arc, log_prob, entropy , h  = test()
# loss = sum(log_prob+entropy)
# print(loss)
# loss.backward()

# arc, log_prob, entropy , h = test()
# loss = sum(log_prob+entropy)
# print(loss)
# loss.backward()
