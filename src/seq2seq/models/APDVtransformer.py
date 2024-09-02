import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils

from seq2seq.modules import MultiheadAttention, LearnedPositionalEmbedding, HierarchicalAttention
from seq2seq.models import Seq2SeqEncoder, Seq2SeqDecoder, Source2TargetModel
from seq2seq.models import register_model, register_model_architecture
    
'''
Training stage2, vocab-level KD
'''

@register_model('APDVtransformer')
class APDVTransformerModel(Source2TargetModel):
    def __init__(self, source_encoder, target_encoder):
        super().__init__(source_encoder, target_encoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, help='dropout probability')
        parser.add_argument('--attention-dropout', default=0.1, type=float, help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', default=0.1, help='dropout probability after ReLU in FFN')
        parser.add_argument('--max-source-positions', default=1024, type=int, help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, help='max number of tokens in the target sequence')

        parser.add_argument('--encoder-embed-path', type=str, help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, help='num encoder attention heads')

        parser.add_argument('--decoder-embed-path', type=str, help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, help='num decoder attention heads')

    @classmethod
    def build_model(cls, args, all_en_dict, src_pt_dict):
        base_architecture(args)

        def build_embedding(dictionary, embed_dim, path=None):
            embedding = nn.Embedding(len(dictionary), embed_dim, padding_idx=dictionary.pad_idx)
            nn.init.kaiming_normal_(embedding.weight)
            nn.init.constant_(embedding.weight[dictionary.pad_idx], 0)
            if path is not None:
                utils.load_embedding(path, dictionary, embedding)
            return embedding

        shared_embed_tokens = build_embedding(all_en_dict, args.decoder_embed_dim, args.decoder_embed_path)
        src_embed_tokens = build_embedding(src_pt_dict, args.decoder_embed_dim, args.decoder_embed_path)

        source_encoder = APDVTransformerEncoder(args, src_pt_dict, src_embed_tokens)
        target_encoder = TransformerEncoder(args, all_en_dict, shared_embed_tokens)
        return APDVTransformerModel(source_encoder, target_encoder)


class APDVTransformerEncoder(Seq2SeqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.dropout = args.dropout
        #self.embed_scale = math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(args.max_source_positions + self.padding_idx + 1, embed_dim, self.padding_idx)
        nn.init.kaiming_normal_(self.embed_positions.weight)
        nn.init.constant_(self.embed_positions.weight[self.padding_idx], 0)
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(args) for _ in range(args.encoder_layers)])

    def forward(self, src_tokens, src_lengths):
        # Embed tokens and positions
        #x = self.embed_scale * self.embed_tokens(src_tokens)
        x = self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Encoder layer
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
    
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

class TransformerEncoder(Seq2SeqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.dropout = args.dropout
        #self.embed_scale = math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(args.max_source_positions + self.padding_idx + 1, embed_dim, self.padding_idx)
        nn.init.kaiming_normal_(self.embed_positions.weight)
        nn.init.constant_(self.embed_positions.weight[self.padding_idx], 0)
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(args) for _ in range(args.encoder_layers)])
        
        for p in self.parameters():
            p.requires_grad = False
    

    def forward(self, encoder_out, src_tokens, src_lengths):
        # Embed tokens and positions
        #x = self.embed_scale * self.embed_tokens(src_tokens)
        x = self.embed_tokens(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
    
        
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # Compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # Encoder layer
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
    
        adapt_pool = nn.AdaptiveAvgPool1d(encoder_out['encoder_out'].size()[0]) 
        x = adapt_pool(x.transpose(0,2)).transpose(0,2)

        return x, encoder_out['encoder_out']

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout

        self.self_attn = MultiheadAttention(self.embed_dim, args.encoder_attention_heads, args.attention_dropout)
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, encoder_padding_mask):
    
        residual = x
        x = self.layer_norms[0](x)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
    
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
     
        residual = x
        x = self.layer_norms[1](x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
    
        return x


@register_model_architecture('APDVtransformer', 'APDVtransformer')
def base_architecture(args):
    args.max_source_positions = getattr(args, 'max_source_positions', 1024)
    args.max_target_positions = getattr(args, 'max_target_positions', 1024)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 0) 
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 0) 
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
