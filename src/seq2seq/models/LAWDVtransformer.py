import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
from torch.autograd import Function
from numba import cuda
from seq2seq import utils

from seq2seq.modules import MultiheadAttention, LearnedPositionalEmbedding, HierarchicalAttention
from seq2seq.models import Seq2SeqEncoder, Seq2SeqDecoder, Source2TargetModel
from seq2seq.models import register_model, register_model_architecture
    
'''
Training stage2, vocab-level KD
'''

@register_model('LAWDVtransformer')
class LAWDVTransformerModel(Source2TargetModel):
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

        source_encoder = LAWDVTransformerEncoder(args, src_pt_dict, src_embed_tokens)
        target_encoder = TransformerEncoder(args, all_en_dict, shared_embed_tokens)
        return LAWDVTransformerModel(source_encoder, target_encoder)


class LAWDVTransformerEncoder(Seq2SeqEncoder):
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
        
        #self.lawd_cost = SoftDTW(gamma=0.1)
        self.lawd_cost = SoftDTW()
        
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

        cost = self.lawd_cost(encoder_out['encoder_out'].transpose(0,1), x.transpose(0,1))

        return cost

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

class SoftDTW(torch.nn.Module):
    def __init__(self, use_cuda=True, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        super(SoftDTW, self).__init__()
        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, x, y):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        bx, lx, dx = x.shape
        by, ly, dy = y.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert dx == dy  # Equal feature dimensions

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply 

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        dis = torch.pow(x - y, 2).sum(3)
        
        dis_n = (dis - torch.min(dis)) / (torch.max(dis) - torch.min(dis))
        dis_n = dis_n.squeeze(0)
        
        return dis

    def forward(self, X, Y):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
     
        func_dtw = self._get_func_dtw(X, Y)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D_xy = self.dist_func(X, Y)
            return func_dtw(D_xy, self.gamma, self.bandwidth)
            
class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, R.clone(), gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        dev = grad_output.device
        dtype = grad_output.dtype
        D, R, gamma, bandwidth = ctx.saved_tensors

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None

@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()
        
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()


@register_model_architecture('LAWDVtransformer', 'LAWDVtransformer')
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
