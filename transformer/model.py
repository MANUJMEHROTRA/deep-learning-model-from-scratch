import math
import torch
import torch.nn as nn 

class InputEmbedding(nn.Module):
    def __init__(self, d_model,vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self,seq_len,d_model,dropout):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(self.seq_len,self.d_model)
        position = torch.arange(0,seq_len,dtype=torch.float32) #shape [seq_len]
        position = position.unsqueeze(1) # doing it for broadcasting the operation [seq_len,1]
        div_term = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0))/self.d_model) # [d_model//2]
        pe[:,0::2] = torch.sin(position*div_term) # position*div_term  = [seq_len,1] * [seq_len//2] = [seq_len,d_model//2]
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0) # shape = [1,seq_len,d_model]
        self.register_buffer("pe",pe)


    def forward(self,x):
        # x = x + self.pe.requires_grad_(False) # NOTE this is also correct when x is of seq len 512.
        x = x + self.pe[:,:x.size()[1],:].requires_grad_(requires_grad = False) # NOTE above notation is also correct given if the seq lent is of 512, howeve x can be of a variable lenth hen using this formula
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,features,epsilon=10**-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(features)) #multiplicative
        self.beta = nn.Parameter(torch.zeros(features)) # addative term
        self.epsilon = epsilon

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True, unbiased=False)
        return (self.alpha * (x-mean)/(std+self.epsilon))  + self.beta



class FeedForwardBlock(nn.Module):
    def __init__(self,d_model=512,ffn_size=2048,dropout = 0.2 ):
        super().__init__()

        self.linear_1 = nn.Linear(d_model,ffn_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ffn_size,d_model)
        self.relu =nn.ReLU()

    def forward(self,x):
        """x.shape = (bactch,seq_len,d_modle)-->
        after linear_1 --> (bach,seq_len,ffn_size) --> 
        after linear_2 --> (bach,seq_len,d_modle) """
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,head=12,dropout= 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.Q_matrix = nn.Linear(d_model,d_model,bias=False)
        self.K_matrix = nn.Linear(d_model,d_model,bias=False)
        self.V_matrix = nn.Linear(d_model,d_model,bias=False)
        self.wo = nn.Linear(d_model,d_model,bias=False)

      
        self.d_k = d_model//head
        self.h = head

    @staticmethod
    def attention(query,key,value,mask,dropout):
        d_k = query.shape[-1]
        attention_score = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill_(mask==0,-1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return attention_score@value ,attention_score

    def forward(self,q,k,v,mask):
        q = self.Q_matrix(q) #(batch,seq,d_model)
        k = self.K_matrix(k) #(batch,seq,d_model)
        v = self.V_matrix(v) #(batch,seq,d_model)

        q = q.reshape(q.shape[0],q.shape[1],self.h,self.d_k).transpose(2,1) # NOTE  HAVE A DOUBT here
        k = k.reshape(k.shape[0],k.shape[1],self.h,self.d_k).transpose(2,1) # NOTE  HAVE A DOUBT here
        v = v.reshape(v.shape[0],v.shape[1],self.h,self.d_k).transpose(2,1) # NOTE  HAVE A DOUBT here
        x, self.attenion_score = MultiHeadAttention.attention(q,k,v,mask,self.dropout)

        x = x.transpose(1,2).contiguous().view(v.shape[0],-1,self.h*self.d_k)

        return self.wo(x)


class ResidualConnections(nn.Module):
    def __init__(self,featuers,dropout=0.2 ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(featuers)


    def forward(self,x,SubLayer):
        # return x + self.dropout(self.norm(SubLayer(x))) # NOTE  HAVE A DOUBT here
        return x + self.dropout(SubLayer(self.norm(x))) # NOTE  HAVE A DOUBT here


class EncoderBlock(nn.Module):
    def __init__(self, features,self_attention: MultiHeadAttention, dropout: float, ffn: FeedForwardBlock ):
        super().__init__()
        self.self_attention = self_attention
        self.dropout = dropout
        self.ffn = ffn
        self.residula_connection = nn.ModuleList([ResidualConnections(features,dropout) for _ in range(2)])

    def forward(self,x,src_mask):
        x = self.residula_connection[0](x,lambda x: self.self_attention(x,x,x,src_mask) )
        x = self.residula_connection[1](x,lambda x: self.ffn(x))
        return x

class Encoder(nn.Module):
    def __init__(self,features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self,featuers, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, dropout: float, ffn: FeedForwardBlock ):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.residual_connection = nn.ModuleList([ResidualConnections(featuers,dropout) for _ in range(3)])
        self.ffn = ffn

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.residual_connection[0](x,lambda x: self.self_attention(x,x,x,tgt_mask) )
        x = self.residual_connection[1](x,lambda x: self.cross_attention(x,encoder_output,encoder_output,src_mask) )
        x = self.residual_connection[2](x,lambda x: self.ffn(x))
        return x 

class Decoder(nn.Module):
    def __init__(self,features, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.linearProjection = nn.Linear(d_model, vocab_size)

    def forward(self,x):
        return torch.log_softmax( self.linearProjection(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, input_embedding: InputEmbedding, source_embedding: InputEmbedding, src_pos: PositionalEmbedding, tgt_pos: PositionalEmbedding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.source_embedding = source_embedding
        self.src_pos = src_pos
        self.tgt_pos =tgt_pos
        self.proj = proj

    def encode(self,x,src_mask):
        x = self.input_embedding(x)
        x = self.src_pos(x)
        return self.encoder(x,src_mask)

    def decode(self,encoder_output,encoder_mask,x,decoder_mask):
        x = self.source_embedding(x)
        x = self.tgt_pos(x)
        return self.decoder(x,encoder_output,encoder_mask,decoder_mask)

    def project(self,x):
        return self.proj(x)


def build_transform(input_vocab_size: float, output_vocab_size: float, input_seq_len: int=512, head:int=12, output_seq_len: int=512,  N: int = 3, ffn_size: int=2048, dropout: float=0.1, d_model = 768):

    input_embedding = InputEmbedding(d_model,input_vocab_size)
    output_embedding = InputEmbedding(d_model,output_vocab_size)

    inp_pos_embed = PositionalEmbedding(input_seq_len,d_model,dropout)
    tgt_pos = PositionalEmbedding(output_seq_len,d_model,dropout)

    encoder_blocks = []
    for _ in range(N):
        self_attenion = MultiHeadAttention(d_model,head,dropout)
        ffn = FeedForwardBlock(d_model,ffn_size,dropout)
        econder_block = EncoderBlock(d_model,self_attenion,dropout,ffn)
        encoder_blocks.append(econder_block)

    encoder = Encoder(d_model,nn.ModuleList(encoder_blocks))

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model,head,dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model,head,dropout)
        ffn = FeedForwardBlock(d_model,ffn_size,dropout)
        decoder_block = DecoderBlock(d_model,decoder_self_attention_block,decoder_cross_attention_block,dropout,ffn)
        decoder_blocks.append(decoder_block)

    decoder = Decoder(d_model,nn.ModuleList(decoder_blocks))

    proj = ProjectionLayer(d_model,output_vocab_size)

    trans = Transformer(encoder,decoder,input_embedding,output_embedding,inp_pos_embed,tgt_pos,proj)

    for params in trans.parameters():
        if params.dim()>1:
            nn.init.xavier_uniform_(params)
    return trans

