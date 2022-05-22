from numpy.core.arrayprint import printoptions
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Graph_QMixer(nn.Module):
    def __init__(self, args, scheme):
        super(Graph_QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.obs_size = scheme["obs"]["vshape"]

        self.embed_dim = args.mixing_embed_dim
        
        # graph assist   self.input_size, self.output_size, self.dropout, self.alpha, self.device = args
        # self.gat_module = GAT_Module(args = (scheme["state"]["vshape"], scheme["obs"]["vshape"], args.n_agents, 0.2, 0.2))
        # self.gat_module = GAT(scheme["obs"]["vshape"], 64, args.n_agents, 0.2, 0.2, 4)
        self.gat_module = ST_GAT(input_size = scheme["obs"]["vshape"], output_size = args.n_agents, n_nodes = args.n_agents)
        # scheme["obs"]["vshape"], args.n_agents, 0.2, 0.2

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def get_graph_info(self, states, obs_ls, adj_ls):
        # print(obs_ls.shape, adj_ls.shape)
        obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
        adj_ls = adj_ls.reshape(-1, self.n_agents, self.n_agents)
        gat_op = self.gat_module(obs_ls, adj_ls)
        gat_weights = gat_op
        return gat_weights

    def get_weights_info(self, agent_qs, states, obs_ls, adj_ls):
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        graph_weights = self.get_graph_info(states, obs_ls, adj_ls)
        return graph_weights
        

    def forward(self, agent_qs, states, obs_ls, adj_ls):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        obs_ls = obs_ls.reshape(-1, self.n_agents, self.obs_size)
        adj_ls = adj_ls.reshape(-1, self.n_agents, self.n_agents)
        graph_weights = self.get_graph_info(states, obs_ls, adj_ls)
        agent_qs = th.bmm(graph_weights, agent_qs.permute(0, 2, 1)).permute(0, 2, 1)


        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot



class GraphAttentionLayer_hyper(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, hyper_input_size, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.hyper_input_size = hyper_input_size
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # self.W = nn.Parameter(th.empty(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(th.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.W_parameter = nn.Linear(self.hyper_input_size, self.in_features * self.out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, state, h, adj):
        self.W = th.abs(self.W_parameter(state).reshape(-1, self.in_features, self.out_features))
        Wh = th.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*th.ones_like(e)
        attention = th.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = th.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = th.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = th.matmul(Wh, self.a[self.out_features:, :])
        # print(Wh1.shape, Wh2.shape)
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(th.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(th.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = th.matmul(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*th.ones_like(e)
        attention = th.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = th.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = th.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = th.matmul(Wh, self.a[self.out_features:, :])
        # print(Wh1.shape, Wh2.shape)
        # broadcast add
        e = Wh1 + Wh2.permute(0,2,1)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = th.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.softmax(F.log_softmax(x, dim=1), -1)
        return F.log_softmax(x, dim=1)

class GAT_hyper(nn.Module):
    def __init__(self, hyper_input_size, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer_hyper(hyper_input_size, nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer_hyper(hyper_input_size, nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, state, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = th.cat([att(state, x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(state, x, adj))
        return F.softmax(F.log_softmax(x, dim=1), -1)




class ST_GAT(nn.Module):
    def __init__(self, input_size, output_size, n_nodes, heads=4, dropout=0.2, initialize = False):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param out_channels Number of output channels
        :param n_nodes Number of nodes in the graph
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super(ST_GAT, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes
        self.input_size = input_size

        self.n_preds = n_nodes
        lstm1_hidden_size = 32
        lstm2_hidden_size = 64

        # single graph attentional layer with 4 attention heads
        self.gat = GAT(input_size, 64, 3*n_nodes, dropout, dropout, heads)
        # self.gat = GAT(input_size, 64, 20, dropout, dropout, heads)

        # add two LSTM layers
        self.lstm1 = th.nn.LSTM(input_size=self.n_nodes, hidden_size=lstm1_hidden_size, num_layers=1)
       
        self.lstm2 = th.nn.LSTM(input_size=lstm1_hidden_size, hidden_size=lstm2_hidden_size, num_layers=1)

         # 初始化循环神经网络参数
        if initialize:
            for name, param in self.lstm1.named_parameters():
                if 'bias' in name:
                    th.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    th.nn.init.xavier_uniform_(param)

            for name, param in self.lstm1.named_parameters():
                if 'bias' in name:
                    th.nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    th.nn.init.xavier_uniform_(param)

        # fully-connected neural network
        self.linear = th.nn.Linear(lstm2_hidden_size, self.n_nodes*self.n_preds)
        if initialize:
            th.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adj):
        """
        Forward pass of the ST-GAT model
        :param data Data to make a pass on
        :param device Device to operate on
        """
        x, edge_index = x, adj

        # gat layer: output of gat: [11400, 12]
        x = self.gat(x, edge_index)
        # x = F.dropout(x, self.dropout, training=self.training)

        # RNN: 2 LSTM
        # [batchsize*n_nodes, seq_length] -> [batch_size, n_nodes, seq_length]
        batch_size = x.shape[0]
        n_node = x.shape[1]
        x = th.reshape(x, (batch_size, n_node, x.shape[2]))
        # for lstm: x should be (seq_length, batch_size, n_nodes)
        # sequence length = 12, batch_size = 50, n_node = 228
        x = x.permute(2, 0, 1)
        # [12, 50, 228] -> [12, 50, 32]
        x, _ = self.lstm1(x)
        # [12, 50, 32] -> [12, 50, 128]
        x, _ = self.lstm2(x)

        # Output contains h_t for each timestep, only the last one has all input's accounted for
        # [12, 50, 128] -> [50, 128]

        # if you only want last output
        x = th.squeeze(x[-1, :, :])
        # [50, 128] -> [50, 228*9]
        x = self.linear(x)

        # Now reshape into final output
        s = x.shape
        # [50, 228*9] -> [50, 228, 9]
        x = F.softmax(th.reshape(x, (s[0], self.n_nodes, self.n_preds)), -1)

        # [50, 228, 9] ->  [11400, 9]
        # 
        # x = th.reshape(x, (s[0]*self.n_nodes, self.n_preds))
        return x















if __name__ == "__main__":
    a = th.randn(2752, 8, 85)
    hyper_inputs = th.randn(32, 60, 120)
    adj = th.ones(a.shape[0], a.shape[1], a.shape[1])
    model = GAT(85, 32, 8, 0.6, 0.2, 4)
    c = model(a, adj)
    print(c.shape)