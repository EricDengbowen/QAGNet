import torch
from torch import nn
from detectron2.utils.registry import Registry
from detectron2.layers import Linear
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
# RELATION_REGISTRY = Registry("RELATION_HEAD")
#
# def build_relation_head(cfg):
#     relation_head_name = cfg.MODEL.RELATION_HEAD.NAME
#     relation_head = RELATION_REGISTRY.get(relation_head_name)(cfg)
#     return relation_head

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class RelationBetweenMulti(nn.Module):
    def __init__(self, unit_nums, in_channels):
        super(RelationBetweenMulti, self).__init__()

        self.inter_channels=in_channels

        self.g = Linear(self.inter_channels, self.inter_channels)
        nn.init.normal_(self.g.weight, mean=0, std=0.01)
        nn.init.constant_(self.g.bias, 0)

        self.theta = Linear(self.inter_channels, self.inter_channels)
        nn.init.normal_(self.theta.weight, mean=0, std=0.01)
        nn.init.constant_(self.theta.bias, 0)

        self.phi = Linear(self.inter_channels, self.inter_channels)
        nn.init.normal_(self.phi.weight, mean=0, std=0.01)
        nn.init.constant_(self.phi.bias, 0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )
        nn.init.normal_(self.concat_project[0].weight, mean=0, std=0.01)

    def forward(self, x):
        g_x = self.g(x)  # 100 256

        theta_x = self.theta(x)
        theta_x = theta_x.permute(1, 0)  # 256 100

        N = theta_x.size(1)
        C = theta_x.size(0)

        theta_x = theta_x.view(C, N, 1)
        theta_x = theta_x.repeat(1, 1, N)  # 256 100 100

        phi_x = self.phi(x)
        phi_x = phi_x.permute(1, 0)  # 256 100
        phi_x = phi_x.view(C, 1, N)
        phi_x = phi_x.repeat(1, N, 1)  # 256 100 100

        concat_feature = torch.cat((theta_x, phi_x), dim=0)  # 512 100 100
        concat_feature = concat_feature.view(1, *concat_feature.size()[:])  # 1 512 100 100

        f = self.concat_project(concat_feature)  # 1 1 100 100
        f = f.view(N, N)  # 100 100
        f_dic_C = f / N

        z = torch.matmul(f_dic_C, g_x)  # 100 256
        return z

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # 节点表示向量的输入特征维度
        self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout  # dropout参数
        self.alpha = alpha  # leakyrelu激活的参数
        self.concat = concat  # 如果为true, 再进行elu激活
        self.device = torch.device("cuda")
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(self.device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1))).to(self.device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # xavier初始化

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [N, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵 维度[N, N] 非零即一，数据结构基本知识
        """
        h = torch.mm(inp, self.W)  # [N, out_features]
        N = h.size()[0]  # N 图的节点数

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        #100,128  100,12800 view-> 10000,128
        #10000,128
        #cat 10000,256
        #view 100,100,256
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        #100,100
        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#@RELATION_REGISTRY.register()
class RelationHead(torch.nn.Module):
    def __init__(self, cfg):
        super(RelationHead, self).__init__()
        inter_channels=cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        unit_nums = cfg.MODEL.RELATION_HEAD.UNIT_NUMS
        self.simpleaverage=False
        self.samescaleg=True
        self.differentscaleg=True

        self.device=torch.device("cuda")
        if self.simpleaverage:
            self.relation_units = []
            self.adj=torch.ones(100,100).to(self.device)
            for idx in range(unit_nums):
                relation_unit = 'relation_unit{}'.format(idx)
                #self.add_module(relation_unit, RelationBetweenMulti(unit_nums,inter_channels))
                self.add_module(relation_unit, GraphAttentionLayer(inter_channels,inter_channels,0.2,0.2,concat=False))
                self.relation_units.append(relation_unit)
            self.fc = Linear(inter_channels*unit_nums, inter_channels) #2048
            nn.init.normal_(self.fc.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc.bias, 0)
        else:
            self.num_query=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES


            if self.samescaleg:

                layer = GraphAttentionLayer(inter_channels, inter_channels, 0.2, 0.2, concat=False)

                #construct same scale subgraph-fully connected:

                # self.ssc_subg_unit_l0 = []
                # self.ssc_subg_unit_l1 = []
                # self.ssc_subg_unit_l2 = []
                #3 different level
                for level in range(3):
                    for idx in range(self.num_query):
                        ssc_subg_units='ssc_subg_unit_level{}_i{}'.format(level,idx)
                        self.add_module(ssc_subg_units,GraphAttentionLayer(inter_channels, inter_channels, 0.2, 0.2, concat=False))
                        #getattr(self, f"ssc_subg_unit_l{level}").append(ssc_subg_units)

                #construct same scale subgraph-directed: nodes -> representative

                # self.ssc_subg_n2r_unit_l0=[]
                # self.ssc_subg_n2r_unit_l1=[]
                # self.ssc_subg_n2r_unit_l2=[]
                #self.adj_ssc_subg_n2r=torch.cat([torch.zeros(4,3),torch.ones(4,1)],dim=1)
                for level in range(3):
                    for idx in range(self.num_query):
                        ssc_subg_n2r_units='ssc_subg_n2r_unit_level{}_i{}'.format(level,idx)
                        self.add_module(ssc_subg_n2r_units,GraphAttentionLayer(inter_channels, inter_channels, 0.2, 0.2, concat=False))
                        # getattr(self, f"ssc_subg_n2r_unit_l{level}").append(ssc_subg_n2r_units)

                #construct different scale subgraph-fully connected
                for idx in range(self.num_query):
                    dsc_subg_units='dsc_subg_unit_i{}'.format(idx)
                    self.add_module(dsc_subg_units,GraphAttentionLayer(inter_channels, inter_channels, 0.2, 0.2, concat=False))

                for idx in range(self.num_query):
                    dsc_subg_n2r_units='dsc_subg_n2r_units_i{}'.format(idx)
                    self.add_module(dsc_subg_n2r_units,GraphAttentionLayer(inter_channels, inter_channels, 0.2, 0.2, concat=False))

                #construct multi instance graph-fully connected
                self.multiinstancegraph=[]
                for idx in range(8):
                    mig_units='mig_units_{}'.format(idx)
                    self.add_module(mig_units,GraphAttentionLayer(inter_channels, inter_channels, 0.2, 0.2, concat=False))
                    self.multiinstancegraph.append(mig_units)
                self.fc = Linear(inter_channels * unit_nums, inter_channels)  # 2048
                nn.init.normal_(self.fc.weight, mean=0, std=0.01)
                nn.init.constant_(self.fc.bias, 0)
        self.saliency_score = Linear(inter_channels, 1)
        weight_init.c2_xavier_fill(self.saliency_score)

        self.class_embed = nn.Linear(inter_channels, 2)
        self.mask_embed = MLP(inter_channels, inter_channels, inter_channels, 3)
        self.decoder_norm = nn.LayerNorm(inter_channels)
    def forward(self, query,mask_features):
        if self.simpleaverage:
            # query_fea=outputs['query'][-1]
            # query_fea=query_fea.permute(1,0,2)
            query_fea = query
            # query_fea = query_fea.permute(1, 0, 2)
            bs = query_fea[-1].size()[1]
            new_query_fea = [f.permute(1, 0, 2) for f in query_fea]
            input_feature = torch.zeros(bs, 100, 256).to(self.device)
            for f in new_query_fea:
                input_feature = input_feature + f
            input_feature = input_feature / bs

            z = []
            scores = []
            for query_fea_per_img in input_feature:
                # result = tuple([getattr(self, relation_uint)(query_fea_per_img,self.adj)
                #                 for relation_uint in self.relation_units])
                # y = torch.cat(result, dim=1)
                # y = self.fc(y)
                # r=F.relu(query_fea_per_img + y)
                # z.append(r)
                # score=self.saliency_score(r)
                # scores.append(score)
                score = self.saliency_score(query_fea_per_img)
                scores.append(score)
            return z, scores
        else:
            query_fea = [f.permute(1, 0, 2) for f in query]  # Nine （2,100,256）
            # same scale features
            qf_level0 = []
            qf_level1 = []
            qf_level2 = []

            bs, num_ins, hdim = query_fea[0].size()  # in resnet50 num_ins:100 hdim:256

            for i, f in enumerate(query_fea):
                if (i % 3) == 0:
                    qf_level0.append(f)
                elif (i % 3) == 1:
                    qf_level1.append(f)
                elif (i % 3) == 2:
                    qf_level2.append(f)
            qf_l0 = torch.stack(qf_level0, dim=1)  # bs,num of query in same level,instan_num,hdim  -> 2,3,100,256
            qf_l1 = torch.stack(qf_level1, dim=1)
            qf_l2 = torch.stack(qf_level2, dim=1)

            qf_all_level = torch.stack([qf_l0, qf_l1, qf_l2],
                                       dim=1)  # bs,level num,num of query in same level,instan_num,hdim  -> 2,3,3,100,256

            # samescaleg representative initialization
            r0 = (sum(qf_level0) / len(qf_level0))  # (bs,100,256)
            r1 = (sum(qf_level1) / len(qf_level1))
            r2 = (sum(qf_level2) / len(qf_level2))
            r_sg_all_level = torch.stack([r0, r1, r2], dim=1)  # (bs,3,100,256)

            r_ins = torch.mean(r_sg_all_level, dim=1)  # from (2,3,100,256) to (2,100,256)

            adj_ssc_subg = torch.ones(3, 3).to(self.device)
            adj_ssc_subg_n2r = torch.cat([torch.zeros(4, 3), torch.ones(4, 1)], dim=1).to(self.device)

            # for level in range(3):
            #     for b,query_fea_sg_per_img in enumerate(eval(f"qf_level{level}")): #(3,100,256)
            #         for i in range(num_ins):
            #             sgdata=query_fea_sg_per_img[:,i,:]#(3,256)
            #             result_sg=getattr(self, f"ssc_subg_unit_level{level}_i{i}")(sgdata,adj_ssc_subg)#(3,256)
            #             sg_w_r=torch.cat([result_sg,eval(f"r{level}")[b,i,:].unsqueeze(0)],dim=0)#(4,256)
            #             result_n2r=getattr(self, f"ssc_subg_n2r_unit_level{level}_i{i}")(sg_w_r,adj_ssc_subg_n2r)#(4,256)
            #             eval(f'r{level}')[b,i]=result_n2r[-1]

            for b, query_fea_all_level_sg_per_img in enumerate(qf_all_level):  # 3,3,100,256
                for level in range(3):
                    for i in range(num_ins):
                        sgdata = qf_all_level[b, level, :, i, :]  # 3,256
                        result_sg = getattr(self, f"ssc_subg_unit_level{level}_i{i}")(sgdata, adj_ssc_subg)  # (3,256)
                        qf_all_level[b, level, :, i, :] = result_sg
                        sg_w_r = torch.cat([result_sg, r_sg_all_level[b, level, i, :].unsqueeze(0)], dim=0)  # (4,256)
                        result_n2r = getattr(self, f"ssc_subg_n2r_unit_level{level}_i{i}")(sg_w_r,
                                                                                           adj_ssc_subg_n2r)  # (4,256)
                        r_sg_all_level[b, level, i, :] = result_n2r[-1]

            # different scale graph
            # instance represetation initialization

            for b, r_sg_all_level_per_img in enumerate(r_sg_all_level):  # (3,100,256)
                for i in range(num_ins):
                    sgdata = r_sg_all_level[b, :, i, :]
                    result_sg = getattr(self, f"dsc_subg_unit_i{i}")(sgdata, adj_ssc_subg)  # (3,256)
                    r_sg_all_level[b, :, i, :] = result_sg
                    sg_w_r = torch.cat([result_sg, r_ins[b, i, :].unsqueeze(0)], dim=0)  # (4,256)
                    result_n2r = getattr(self, f"dsc_subg_n2r_units_i{i}")(sg_w_r, adj_ssc_subg_n2r)  # (4,256)
                    r_ins[b, i, :] = result_n2r[-1]

            adj_multi_ins = torch.ones(100, 100).to(self.device)

            z = []
            scores = []
            for r_ins_perimage in r_ins:  # (100,256)
                result = tuple([getattr(self, multi_ins_unit)(r_ins_perimage, adj_multi_ins)
                                for multi_ins_unit in self.multiinstancegraph])
                y = torch.cat(result, dim=1)
                y = self.fc(y)
                r = F.relu(r_ins_perimage + y)
                z.append(r)
                score = self.saliency_score(r)
                scores.append(score)
            z_b=torch.stack(z,dim=1)
            outputs_class, outputs_mask=self.forward_prediction_heads(z_b,mask_features)
            return scores, outputs_class, outputs_mask

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)


        return outputs_class, outputs_mask





