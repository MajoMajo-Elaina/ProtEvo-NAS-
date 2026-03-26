import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F


class inter_model(nn.Module):
    def __init__(self, input_size, hidden_size, architecture):
        super(inter_model, self).__init__()
        self.act1_type = 'relu'
        self.act2_type = 'relu'
        self.embedding_layer = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)
        self.activation_dict = {'relu': nn.ReLU(),
                                'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
                                'prelu': nn.PReLU(num_parameters=1),
                                'rrelu': nn.RReLU(lower=0.1, upper=0.5),
                                'elu': nn.ELU(alpha=1.0),
                                'selu': nn.SELU(),
                                'celu': nn.CELU(alpha=1.0),
                                'sigmoid': nn.Sigmoid(),
                                'tanh': nn.Tanh(),
                                'relu6': nn.ReLU6(),
                                'softplus': nn.Softplus(beta=1, threshold=20)
                                }
        self.act1 = self.activation_dict.get(
            self.act1_type,
            nn.ReLU()
        )
        self.act2 = self.activation_dict.get(
            self.act2_type,
            nn.ReLU()
        )
        self.linearLayer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            self.act1,
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            self.act2
        )

    def forward(self, inter_feature):
        inter_feature = F.relu(self.embedding_layer(*inter_feature))
        inter_feature = self.linearLayer(inter_feature)
        return inter_feature


class transformer_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, head=1):
        super(transformer_block, self).__init__()
        self.head = head
        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(head)])
        self.concat_trans = nn.Linear((hidden_dim) * head, hidden_dim, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.layernorm = nn.LayerNorm(in_dim)

    def forward(self, g, residue_h, inter_h):
        multi_output = []
        for i in range(self.head):
            q = self.trans_q_list[i](residue_h)
            k = self.trans_k_list[i](inter_h)
            v = self.trans_v_list[i](residue_h)
            att = torch.sum(torch.mul(q, k) / torch.sqrt(torch.tensor(1280.0)), dim=1, keepdim=True)
            with g.local_scope():
                g.ndata['att'] = att.reshape(-1)
                alpha = dgl.softmax_nodes(g, 'att').reshape((v.size(0), 1))
                tp = v * alpha
            multi_output.append(tp)
        multi_output = torch.cat(multi_output, dim=1)
        multi_output = self.concat_trans(multi_output)
        multi_output = self.layernorm(multi_output + residue_h)
        multi_output = self.layernorm(self.ff(multi_output) + multi_output)
        return multi_output


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, head, architecture):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv1_type = architecture[0]
        self.conv2_type = architecture[2]
        self.act1_config = architecture[1] 
        self.act2_config = architecture[3]  
        self.pooling_method = architecture[7]
        self.activation_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
            'prelu': nn.PReLU(num_parameters=1),
            'rrelu': nn.RReLU(lower=0.1, upper=0.5),
            'elu': nn.ELU(alpha=1.0),
            'selu': nn.SELU(),
            'celu': nn.CELU(alpha=1.0),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu6': nn.ReLU6(),
            'softplus': nn.Softplus(beta=1, threshold=20)
        }
        self.act1 = self.activation_dict.get(
            self.act1_config,
            nn.ReLU()
        )
        self.act2 = self.activation_dict.get(
            self.act2_config,
            nn.ReLU()
        )
        self.transformer_block = transformer_block(hidden_dim, hidden_dim, head)

        if self.conv1_type == 'GCN':
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        elif self.conv1_type == 'GAT':
            self.conv1 = dglnn.GATConv(in_dim, hidden_dim // head, num_heads=head, feat_drop=0.3,
                                       attn_drop=0.3,
                                       residual=True,
                                       activation=None)
        elif self.conv1_type == 'ChebConv':
            self.conv1 = dglnn.ChebConv(in_dim, hidden_dim, k=3)

        elif self.conv1_type == 'SAGEConv':
            self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, 'mean')
        else:
            raise ValueError(f"Unsupported conv1 type: {conv1_type}")

        if self.conv2_type == 'GCN':
            self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        elif self.conv2_type == 'GAT':
            self.conv2 = dglnn.GATConv(hidden_dim, hidden_dim // head, num_heads=head, feat_drop=0.3,
                                       attn_drop=0.3,
                                       residual=True,
                                       activation=None)
        elif self.conv2_type == 'ChebConv':
            self.conv2 = dglnn.ChebConv(hidden_dim, hidden_dim, k=3)

        elif self.conv2_type == 'SAGEConv':
            self.conv2 = dglnn.SAGEConv(hidden_dim, hidden_dim, 'mean')
        else:
            raise ValueError(f"Unsupported conv2 type: {conv2_type}")

    def forward(self, g, h, inter_f):
        with g.local_scope():
            g.ndata['h'] = h
            if self.pooling_method == 'mean':
                init_avg_h = dgl.mean_nodes(g, 'h')
            elif self.pooling_method == 'sum':
                init_avg_h = dgl.sum_nodes(g, 'h')
            elif self.pooling_method == 'max':
                init_avg_h = dgl.max_nodes(g, 'h')
        pre = h
        h = self.bn1(h)
        h_conv1 = self.conv1(g, h)
        if isinstance(self.conv1, dglnn.GATConv):
            h_conv1 = h_conv1.flatten(1)  
        h = pre + self.dropout(self.act1(h_conv1))

        pre = h
        h = self.bn2(h)
        h_conv2 = self.conv2(g, h)
        if isinstance(self.conv2, dglnn.GATConv):
            h_conv2 = h_conv2.flatten(1)
        h = pre + self.dropout(self.act2(h_conv2))

        with g.local_scope():
            g.ndata['inter'] = dgl.broadcast_nodes(g, inter_f)
            residue_h = h
            inter_h = g.ndata['inter']
            hg = self.transformer_block(g, residue_h, inter_h)
            g.ndata['output'] = hg
            readout = dgl.sum_nodes(g, "output")
            return readout, init_avg_h


class combine_inter_model(nn.Module):
    def __init__(self, inter_size, graph_size, label_num, architecture):
        super(combine_inter_model, self).__init__()
        self.inter_hid = 1280
        self.graph_hid = 1280
        self.head = int(architecture[4])
        self.act1_type = architecture[5]
        self.act2_type = architecture[6]
        self.activation_dict = {'relu': nn.ReLU(),
                                'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
                                'prelu': nn.PReLU(num_parameters=1),
                                'rrelu': nn.RReLU(lower=0.1, upper=0.5),
                                'elu': nn.ELU(alpha=1.0),
                                'selu': nn.SELU(),
                                'celu': nn.CELU(alpha=1.0),
                                'sigmoid': nn.Sigmoid(),
                                'tanh': nn.Tanh(),
                                'relu6': nn.ReLU6(),
                                'softplus': nn.Softplus(beta=1, threshold=20)
                                }
        self.inter_embedding = inter_model(inter_size, self.inter_hid, architecture)

        self.GNN = GCN(graph_size, self.graph_hid, label_num, self.head, architecture)
        self.act1 = self.activation_dict.get(
            self.act1_type,
            nn.ReLU()
        )
        self.act2 = self.activation_dict.get(
            self.act2_type,
            nn.ReLU()
        )

        self.classify = nn.Sequential(
            nn.BatchNorm1d(graph_size + self.graph_hid),
            nn.Linear(graph_size + self.graph_hid, (graph_size + self.graph_hid) * 2),
            nn.Dropout(0.3),
            self.act1,
            nn.Linear((graph_size + self.graph_hid) * 2, (graph_size + self.graph_hid) * 2),
            nn.Dropout(0.3),
            self.act2,
            nn.Linear((graph_size + self.graph_hid) * 2, label_num)
        )

    def forward(self, inter_feature, graph, graph_h):
        inter_feature = self.inter_embedding(inter_feature)
        graph_feature, init_feature = self.GNN(graph, graph_h, inter_feature)
        return self.classify(torch.cat((init_feature, graph_feature), 1))







