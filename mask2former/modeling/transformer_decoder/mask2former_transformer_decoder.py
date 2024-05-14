
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from ..relation import GraphAttentionLayer
from detectron2.layers import Linear


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        relation_layer_num: int,
        unit_num: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # ------------------------QAGNet------------------------

        self.graph_headnum=unit_num
        self.siderankloss=True
        self.relation_layer_num = relation_layer_num
        self.relation_sscg_n2n=nn.ModuleList()
        self.relation_sscg_n2r=nn.ModuleList()
        self.relation_dscg_n2n=nn.ModuleList()
        self.relation_dscg_n2r=nn.ModuleList()

        self.relation_multi_ins_g=nn.ModuleList()

        self.relation_dscg_r2n = nn.ModuleList()
        self.relation_sscg_r2n=nn.ModuleList()

        self.relation_multi_ins_g_final=nn.ModuleList()

        #N2R Nodes to Representative
        for i in range(self.relation_layer_num+1):
            # ssc (Single Scale)
            for level in range(3):
                ssc_subg_n2n_units = 'ssc_subg_n2n_unit_layer{}_level{}'.format(i,level)
                self.relation_sscg_n2n.add_module(ssc_subg_n2n_units, GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))
                ssc_subg_n2r_units = 'ssc_subg_n2r_unit_layer{}_level{}'.format(i,level)
                self.relation_sscg_n2r.add_module(ssc_subg_n2r_units,GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))

            #dsc (Different Scale)
            dsc_subg_n2n_units = 'dsc_subg_n2n_unit_layer{}'.format(i)
            self.relation_dscg_n2n.add_module(dsc_subg_n2n_units, GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))
            dsc_subg_n2r_units = 'dsc_subg_n2r_unit_layer{}'.format(i)
            self.relation_dscg_n2r.add_module(dsc_subg_n2r_units,GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))


        #R2N Representative to Nodes

        for i in range(self.relation_layer_num):
            #multi_ins
            mig_units='mig_units_layer{}'.format(i)
            self.relation_multi_ins_g.add_module(mig_units,GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))
            #dscg2sscg
            for level in range(3):
                ssc_subg_r2n_units=  'ssc_subg_r2n_unit_layer{}_level{}'.format(i,level)
                self.relation_sscg_r2n.add_module(ssc_subg_r2n_units,GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))
            #Ins2dscg
            dsc_subg_r2n_units='dsc_subg_r2n_unit_layer{}'.format(i)
            self.relation_dscg_r2n.add_module(dsc_subg_r2n_units,GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))

        #multi_ins_final
        for idx in range(self.graph_headnum):
            mig_units_final = 'mig_units_final{}'.format(idx)
            self.relation_multi_ins_g_final.add_module(mig_units_final,GraphAttentionLayer(hidden_dim, hidden_dim, 0.2, 0.2, concat=True))

        self.relation_multifc = Linear(hidden_dim * unit_num, hidden_dim)  # 2048
        nn.init.xavier_uniform_(self.relation_multifc.weight)
        nn.init.constant_(self.relation_multifc.bias, 0)


        self.relation_saliency_score = Linear(hidden_dim,1)
        nn.init.xavier_uniform_(self.relation_saliency_score.weight)
        nn.init.constant_(self.relation_saliency_score.bias, 0)


    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["relation_layer_num"] = cfg.MODEL.RELATION_HEAD.LAYER_NUM
        ret["unit_num"]= cfg.MODEL.RELATION_HEAD.UNIT_NUMS

        return ret

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        query=[]

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            query.append(output)
            outputs_class, outputs_mask, attn_mask= self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        ##----------QAGNet----------##


        query_fea = [f.permute(1, 0, 2) for f in query]

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

        qf_l0 = torch.stack(qf_level0, dim=1)  # bs,num of query in same level,instan_num,hdim
        qf_l1 = torch.stack(qf_level1, dim=1)
        qf_l2 = torch.stack(qf_level2, dim=1)
        qf_all_level = torch.stack([qf_l0, qf_l1, qf_l2],dim=1)  # bs,level num, num of query in same level,instan_num,hdim

        # representative initialization

        r_sg_all_level = torch.mean(qf_all_level, dim=2)
        r_ins = torch.mean(r_sg_all_level, dim=1)

        # adj matrix
        adj_n2n = torch.eye(num_ins)
        adj_n2n = (adj_n2n.repeat(3, 3)).to("cuda")

        adj_n2r = torch.eye(num_ins * 3, num_ins * 3)
        m = torch.zeros(num_ins, num_ins).repeat(3, 1)
        adj_n2r = torch.cat([adj_n2r, m], dim=1)
        md = torch.eye(num_ins).repeat(1, 4)
        adj_n2r = torch.cat([adj_n2r, md], dim=0).to("cuda")


        adj_r2n = torch.eye(num_ins * 4, num_ins * 3)
        m = torch.eye(num_ins).repeat(4, 1)
        adj_r2n = torch.cat([adj_r2n, m], dim=1).to("cuda")

        adj_multi_ins = torch.ones(num_ins, num_ins).to("cuda")
        if self.siderankloss:
            side_rank_score=[]

        preInsFea = []

        for layer_num in range(self.relation_layer_num):

            for b, query_fea_all_level_sg_per_img in enumerate(qf_all_level):
                for level in range(3):
                    sgdata = qf_all_level[b, level, :, :, :]
                    sgdata = sgdata.view(num_ins * 3, hdim)
                    result_sg = getattr(self.relation_sscg_n2n, f"ssc_subg_n2n_unit_layer{layer_num}_level{level}")(sgdata, adj_n2n)
                    qf_all_level[b, level, :, :, :] = result_sg.reshape(3, num_ins, hdim)
                    sg_w_r = torch.cat([result_sg, r_sg_all_level[b, level, :, :]], dim=0)
                    result_n2r = getattr(self.relation_sscg_n2r,f"ssc_subg_n2r_unit_layer{layer_num}_level{level}")(sg_w_r,adj_n2r)
                    r_sg_all_level[b, level, :, :] = result_n2r[num_ins * 3:, :]

            for b, r_sg_all_level_per_img in enumerate(r_sg_all_level):
                sgdata = r_sg_all_level[b, :, :, :]
                sgdata = sgdata.view(num_ins * 3, hdim)
                result_sg = getattr(self.relation_dscg_n2n, f"dsc_subg_n2n_unit_layer{layer_num}")(sgdata,adj_n2n)

                r_sg_all_level[b, :, :, :] = result_sg.reshape(3, num_ins, hdim)
                sg_w_r = torch.cat([result_sg, r_ins[b, :, :]], dim=0)
                result_n2r = getattr(self.relation_dscg_n2r, f"dsc_subg_n2r_unit_layer{layer_num}")(sg_w_r,adj_n2r)

                r_ins[b, :, :] = result_n2r[num_ins * 3:, :]

            # -----short connection-----
            preInsFea.append(r_ins)
            if layer_num > 0:
                r_ins = F.elu(r_ins + preInsFea[-2])
            if self.siderankloss:
                side_rank_score_per_layer=[]
            for b, r_ins_per_image in enumerate(r_ins):
                # Multi Instance Graph Node Updating
                r_ins_data = r_ins[b, :, :]
                result_r_ins = getattr(self.relation_multi_ins_g, f"mig_units_layer{layer_num}")(r_ins_data,adj_multi_ins)
                r_ins[b, :, :] = result_r_ins
                if self.siderankloss:
                    side_rank_score_per_layer.append(self.relation_saliency_score(r_ins[b, :, :]))

            if self.siderankloss:
                side_rank_score.append(side_rank_score_per_layer)

            for b, r_sg_all_level_per_img in enumerate(r_sg_all_level):
                sgdata = r_sg_all_level[b, :, :, :]
                sgdata = sgdata.view(num_ins * 3, hdim)
                sg_w_r = torch.cat([sgdata, r_ins[b, :, :]], dim=0)
                result_r2n = getattr(self.relation_dscg_r2n, f"dsc_subg_r2n_unit_layer{layer_num}")(sg_w_r,adj_r2n)
                r_sg_all_level[b, :, :, :] = result_r2n[:num_ins*3, :].reshape(3, num_ins, hdim)

            for b, query_fea_all_level_sg_per_img in enumerate(qf_all_level):
                for level in range(3):
                    sgdata = qf_all_level[b, level, :, :, :]
                    sgdata = sgdata.view(num_ins * 3, hdim)
                    sg_w_r = torch.cat([sgdata, r_sg_all_level[b, level, :, :]], dim=0)
                    result_r2n = getattr(self.relation_sscg_r2n,f"ssc_subg_r2n_unit_layer{layer_num}_level{level}")(sg_w_r,adj_r2n)  # (400,256)
                    qf_all_level[b, level, :, :, :] = result_r2n[:num_ins*3, :].reshape(3, num_ins, hdim)


        for b, query_fea_all_level_sg_per_img in enumerate(qf_all_level):
            for level in range(3):
                sgdata = qf_all_level[b, level, :, :, :]
                sgdata = sgdata.view(num_ins * 3, hdim)
                result_sg = getattr(self.relation_sscg_n2n,f"ssc_subg_n2n_unit_layer{self.relation_layer_num}_level{level}")(sgdata,adj_n2n)
                qf_all_level[b, level, :, :, :] = result_sg.reshape(3, num_ins, hdim)
                sg_w_r = torch.cat([result_sg, r_sg_all_level[b, level, :, :]], dim=0)
                result_n2r = getattr(self.relation_sscg_n2r,f"ssc_subg_n2r_unit_layer{self.relation_layer_num}_level{level}")(sg_w_r,adj_n2r)
                r_sg_all_level[b, level, :, :] = result_n2r[num_ins * 3:, :]

        for b, r_sg_all_level_per_img in enumerate(r_sg_all_level):
            sgdata = r_sg_all_level[b, :, :, :]
            sgdata = sgdata.view(num_ins * 3, hdim)
            result_sg = getattr(self.relation_dscg_n2n, f"dsc_subg_n2n_unit_layer{self.relation_layer_num}")(sgdata,adj_n2n)
            r_sg_all_level[b, :, :, :] = result_sg.reshape(3, num_ins, hdim)
            sg_w_r = torch.cat([result_sg, r_ins[b, :, :]], dim=0)
            result_n2r = getattr(self.relation_dscg_n2r, f"dsc_subg_n2r_unit_layer{self.relation_layer_num}")(sg_w_r,adj_n2r)
            r_ins[b, :, :] = result_n2r[num_ins * 3:, :]

        # Final Step
        # -----short connection-----
        r_ins = F.elu(r_ins + preInsFea[-1])
        z = []
        scores = []
        for b,r_ins_per_image in enumerate(r_ins):
            result = tuple([multi_ins_unit(r_ins_per_image, adj_multi_ins) for multi_ins_unit in self.relation_multi_ins_g_final])
            y = torch.cat(result, dim=1)
            y = self.relation_multifc(y)
            r = F.relu(r_ins_per_image + y)
            z.append(r)
            score = self.relation_saliency_score(r)
            scores.append(score)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
        }

        if self.siderankloss:
            return out, scores, side_rank_score
        else:
            return out, scores,

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # (2 100 256) (2 256 256 256) -> (2 100 256 256)
        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
