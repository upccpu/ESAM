import math
import torch.nn as nn
import torch
from images.image_models import ImageEncoder
from text.text_models import TextEncoder, TextEncoder_without_know
from interraction.inter_models import CroModality
import utils.gat as tg_conv
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig
from utils import L2_norm, cosine_distance
from transformers import BertModel
from utils.data_utils import pad_tensor
from utils.pre_model import RobertaEncoder
import copy

def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask

# class Text_Graph_Encoder(nn.Module):
#     def __init__(self, input_size=300, txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False):
#         super(Text_Graph_Encoder, self).__init__()

#         self.input_size = input_size
#         self.txt_gat_layer = txt_gat_layer
#         self.txt_gat_drop = txt_gat_drop
#         self.txt_gat_head = txt_gat_head
#         self.txt_self_loops = txt_self_loops
#         self.norm = nn.LayerNorm(self.input_size)
#         self.relu1 = nn.GELU()

#         self.txt_conv = nn.ModuleList(
#             [tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size, heads=self.txt_gat_head,
#                              concat=False, dropout=self.txt_gat_drop, fill_value="mean",
#                              add_self_loops=self.txt_self_loops, is_text=True)
#              for i in range(self.txt_gat_layer)])

#     def forward(self, t2, edge_index, gnn_mask):
#         # (N,token_length)
#         tnp = t2
#         # for node with out edge, it representation will be zero-vector
        
#         for gat in self.txt_conv:
#             tnp = self.norm(torch.stack([(self.relu1(gat(data[0], data[1].cuda(), mask=data[2]))) for data in zip(tnp, edge_index, gnn_mask)]))
#         #  congruity score of compositional level

#         return tnp

# class Prompt_Encoder(nn.Module):
#     def __init__(self, input_size=300, img_gat_layer=2, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False):
#         super(Prompt_Encoder, self).__init__()
#         self.input_size = input_size
#         self.img_gat_layer = img_gat_layer
#         self.img_gat_drop = img_gat_drop
#         self.img_gat_head = img_gat_head
#         self.img_self_loops = img_self_loops
#         self.img_conv = nn.ModuleList([tg_conv.GATConv(in_channels=self.input_size, out_channels=self.input_size,
#                                                         heads=self.img_gat_head, concat=False,
#                                                         dropout=self.img_gat_drop, fill_value="mean",
#                                                         add_self_loops=self.img_self_loops, is_text=True) for i in
#                                           range(self.img_gat_layer)])
#         self.norm = nn.LayerNorm(self.input_size)
#         self.relu1 = nn.GELU()

# #         # for token compute the importance of each token
# #         self.linear1 = nn.Linear(self.input_size, 1)
# #         # for np compute the importance of each np
# #         self.linear2 = nn.Linear(self.input_size, 1)
# #         self.norm = nn.LayerNorm(self.input_size)
# #         self.relu1 = nn.ReLU()

#     def forward(self, v2, img_edge_index):
#         # prompt graph encoder
#         v3 = v2
#         for gat in self.img_conv:
#             v3 = self.norm(torch.stack([self.relu1(gat(data[0], data[1].cuda())) for data in zip(v3, img_edge_index)]))

#         return v3


class KEHModel_without_know(nn.Module):
    """
    Our model for Image Repurpose Task
    """

    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768, img_inter_dim=500, img_out_dim=300,
                 cro_layers=1, cro_heads=5, cro_drop=0.2,
                 txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False,
                 img_gat_layer=6, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, img_edge_dim=0,
                 img_patch=49, lam=1, type_bmco=0, visualization=False):
        super(KEHModel_without_know, self).__init__()
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size

        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim

        if self.img_out_dim is not self.txt_out_size:
            self.img_out_dim = self.txt_out_size

        self.cro_layers = cro_layers
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.type_bmco = type_bmco

        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.img_edge_dim = img_edge_dim

        if self.img_gat_layer is not self.txt_gat_layer:
            self.img_gat_layer = self.txt_gat_layer
        if self.img_gat_drop is not self.txt_gat_drop:
            self.img_gat_drop = self.txt_gat_drop
        if self.img_gat_head is not self.txt_gat_head:
            self.img_gat_head = self.txt_gat_head

        self.img_patch = img_patch
        
        self.txt_encoder = TextEncoder_without_know(input_size=self.txt_input_dim, out_size=self.txt_out_size)
        
        self.text_config = copy.deepcopy(self.txt_encoder.get_config())    
        self.text_config.num_attention_heads = self.cro_heads
        self.text_config.hidden_size = self.txt_out_size
        self.text_config.num_hidden_layers = 6
        
        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False
        
        
        self.img_encoder = ImageEncoder(input_dim=self.img_input_dim, inter_dim=self.img_inter_dim,
                                        output_dim=self.img_out_dim)
#         self.img_prompt = Prompt_Encoder(input_size=self.img_out_dim, img_gat_layer=2
#                                    , img_gat_drop=self.img_gat_drop, img_gat_head=self.img_gat_head,
#                                    img_self_loops=self.img_self_loops)
        self.text_prompt_encoder = RobertaEncoder(self.text_config)    
        
#         self.text_graph_encoder = Text_Graph_Encoder(input_size=self.img_out_dim, txt_gat_layer=self.txt_gat_layer,
#                                    txt_gat_drop=self.txt_gat_drop,
#                                    txt_gat_head=self.txt_gat_head, txt_self_loops=self.txt_self_loops)
        self.output_attention_text = nn.Linear(self.img_out_dim, 1)
        self.output_attention_image = nn.Linear(self.img_out_dim, 1)
        
        self.att = nn.Linear(self.img_out_dim, 1, bias=False)
        
        self.classifier_fuse = nn.Linear(in_features=self.img_out_dim, out_features=2)
        self.classifier_text = nn.Linear(in_features=self.img_out_dim, out_features=2)
        self.classifier_image = nn.Linear(in_features=self.img_out_dim, out_features=2)

        self.linear_text = nn.Linear(in_features=self.img_out_dim, out_features=self.img_out_dim)
        self.linear_image = nn.Linear(in_features=self.img_out_dim, out_features=self.img_out_dim)
        
        self.visulization = visualization

    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq, txt_edge_index,
                gnn_mask, np_mask, img_edge_attr=None, key_padding_mask_img=None):

        
        t1 = self.txt_encoder(t1=texts, word_seq=t1_word_seq,
                                        key_padding_mask=mask_batch)
        imgs_prompt = self.img_encoder(imgs.to(t1.dtype))
        prompt_mask = torch.ones(imgs_prompt.size()[:-1],dtype=torch.long).to(imgs.device)
        mask_batch = mask_batch.long()
        ones_mask = torch.ones(mask_batch.size(),dtype=torch.long).to(imgs.device)
        mask_batch = torch.abs(mask_batch-ones_mask)
        text_image_cat = torch.cat([imgs_prompt, t1], dim=1)
        text_image_mask = torch.cat([prompt_mask, mask_batch], dim=1)
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_image_cat.size())

        text_image_transformer = self.text_prompt_encoder(text_image_cat,
                                                 attention_mask=extended_attention_mask,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=extended_attention_mask,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_image_transformer = text_image_transformer.last_hidden_state
        
        image_transformer = text_image_transformer[:, :imgs_prompt.size(1), :]
        text_transformer = text_image_transformer[:, imgs_prompt.size(1):, :]     
        
        text_mask = mask_batch.permute(1, 0).contiguous()
        text_mask = text_mask[0:text_transformer.size(1)]
        text_mask = text_mask.permute(1, 0).contiguous()
        text_alpha = self.output_attention_text(text_transformer)
        text_alpha = text_alpha.squeeze(-1).masked_fill(text_mask == 0, -1e9)
        text_alpha = torch.softmax(text_alpha, dim=-1)
        text_output = (text_alpha.unsqueeze(-1) * text_transformer).sum(dim=1)
        
        
        image_mask = prompt_mask.permute(1, 0).contiguous()
        image_mask = image_mask[0:image_transformer.size(1)]
        image_mask = image_mask.permute(1, 0).contiguous()
        image_alpha = self.output_attention_image(image_transformer)
        image_alpha = image_alpha.squeeze(-1).masked_fill(image_mask == 0, -1e9)
        image_alpha = torch.softmax(image_alpha, dim=-1)
        image_output = (image_alpha.unsqueeze(-1) * image_transformer).sum(dim=1)
        
        text_weight = self.att(text_output)
        image_weight = self.att(image_output)    
        att = nn.functional.softmax(torch.stack((text_weight, image_weight), dim=-1),dim=-1)
        tw, iw = att.split([1,1], dim=-1)
        fuse_output = tw.squeeze(1) * text_output + iw.squeeze(1) * image_output
        
        
        
        logits_fuse = self.classifier_fuse(fuse_output)
        logits_text = self.classifier_text(text_output)
        logits_image = self.classifier_image(image_output)
        
    
        image_cl = self.linear_image(image_output) 
        text_cl = self.linear_text(text_output)
        
   
        fuse_score = nn.functional.softmax(logits_fuse, dim=-1)
        text_score = nn.functional.softmax(logits_text, dim=-1)
        image_score = nn.functional.softmax(logits_image, dim=-1)
#         score = fuse_score + text_score + image_score
        score = fuse_score
        
        
        
        if self.visulization:
            return y, a, pv
        else:
            return logits_fuse, logits_text, logits_image, score, image_cl, text_cl 

        
class KEHModel(nn.Module):
    """
    Our model for Image Repurpose Task
    """

    def __init__(self, txt_input_dim=768, txt_out_size=300, img_input_dim=768, img_inter_dim=500, img_out_dim=300,
                 cro_layers=1, cro_heads=5, cro_drop=0.2,
                 txt_gat_layer=2, txt_gat_drop=0.2, txt_gat_head=5, txt_self_loops=False,
                 img_gat_layer=6, img_gat_drop=0.2, img_gat_head=5, img_self_loops=False, img_edge_dim=0,
                 img_patch=49, lam=1, type_bmco=0, visualization=False):
        super(KEHModel_without_know, self).__init__()
        self.txt_input_dim = txt_input_dim
        self.txt_out_size = txt_out_size

        self.img_input_dim = img_input_dim
        self.img_inter_dim = img_inter_dim
        self.img_out_dim = img_out_dim

        if self.img_out_dim is not self.txt_out_size:
            self.img_out_dim = self.txt_out_size

        self.cro_layers = cro_layers
        self.cro_heads = cro_heads
        self.cro_drop = cro_drop
        self.type_bmco = type_bmco

        self.txt_gat_layer = txt_gat_layer
        self.txt_gat_drop = txt_gat_drop
        self.txt_gat_head = txt_gat_head
        self.txt_self_loops = txt_self_loops
        self.img_gat_layer = img_gat_layer
        self.img_gat_drop = img_gat_drop
        self.img_gat_head = img_gat_head
        self.img_self_loops = img_self_loops
        self.img_edge_dim = img_edge_dim

        if self.img_gat_layer is not self.txt_gat_layer:
            self.img_gat_layer = self.txt_gat_layer
        if self.img_gat_drop is not self.txt_gat_drop:
            self.img_gat_drop = self.txt_gat_drop
        if self.img_gat_head is not self.txt_gat_head:
            self.img_gat_head = self.txt_gat_head

        self.img_patch = img_patch
        
        self.txt_encoder = TextEncoder_without_know(input_size=self.txt_input_dim, out_size=self.txt_out_size)
        
        self.text_config = copy.deepcopy(self.txt_encoder.get_config())    
        self.text_config.num_attention_heads = self.cro_heads
        self.text_config.hidden_size = self.txt_out_size
        self.text_config.num_hidden_layers = 6
        
        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False
        
        
        self.img_encoder = ImageEncoder(input_dim=self.img_input_dim, inter_dim=self.img_inter_dim,
                                        output_dim=self.img_out_dim)
        self.img_prompt = Prompt_Encoder(input_size=self.img_out_dim, img_gat_layer=2
                                   , img_gat_drop=self.img_gat_drop, img_gat_head=self.img_gat_head,
                                   img_self_loops=self.img_self_loops)
        self.text_prompt_encoder = RobertaEncoder(self.text_config)    
        
        self.text_graph_encoder = Text_Graph_Encoder(input_size=self.img_out_dim, txt_gat_layer=self.txt_gat_layer,
                                   txt_gat_drop=self.txt_gat_drop,
                                   txt_gat_head=self.txt_gat_head, txt_self_loops=self.txt_self_loops)
        self.output_attention = nn.Linear(self.img_out_dim, 1)
        

        self.linear1 = nn.Linear(in_features=self.img_out_dim, out_features=2)
        self.linear2 = nn.Linear(in_features=self.img_out_dim, out_features=self.img_out_dim)
        
        self.visulization = visualization

    def forward(self, imgs, texts, mask_batch, img_edge_index, t1_word_seq, txt_edge_index,
                gnn_mask, np_mask, img_edge_attr=None, key_padding_mask_img=None):
        
        imgs = self.img_encoder(imgs)
        imgs_prompt = self.img_prompt(imgs, img_edge_index=img_edge_index)
        texts = self.txt_encoder(t1=texts, word_seq=t1_word_seq,
                                        key_padding_mask=mask_batch)
        t1 = self.text_graph_encoder(texts, edge_index=txt_edge_index, gnn_mask=gnn_mask)
        prompt_mask = torch.ones(imgs_prompt.size()[:-1],dtype=torch.long).to(imgs.device)
        mask_batch = mask_batch.long()
        ones_mask = torch.ones(mask_batch.size(),dtype=torch.long).to(imgs.device)
        mask_batch = torch.abs(mask_batch-ones_mask)
        text_image_cat = torch.cat([imgs_prompt, t1], dim=1)
        text_image_mask = torch.cat([prompt_mask, mask_batch], dim=1)
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_image_cat.size())

        text_transformer = self.text_prompt_encoder(text_image_cat,
                                                 attention_mask=extended_attention_mask,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=extended_attention_mask,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        text_transformer = text_transformer.last_hidden_state
        
        text_mask = text_image_mask.permute(1, 0).contiguous()
        text_mask = text_mask[0:text_transformer.size(1)]
        text_mask = text_mask.permute(1, 0).contiguous()
        text_image_alpha = self.output_attention(text_transformer)
        text_image_alpha = text_image_alpha.squeeze(-1).masked_fill(text_mask == 0, -1e9)
        text_image_alpha = torch.softmax(text_image_alpha, dim=-1)
        
        
        text_image_output = (text_image_alpha.unsqueeze(-1) * text_transformer).sum(dim=1)
        
        output = self.linear1(text_image_output)
        y_cl = self.linear2(text_image_output)
        
        if self.visulization:
            return y, a, pv
        else:
            return output, y_cl        