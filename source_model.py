import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

    def __init__(self, d_model : int, vocabulary_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embedding_layer = nn.Embedding(vocabulary_size, d_model)

    def forward(self, x):
        return self.embedding_layer(x)*math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model : int, max_seq_len : int, dropout : int): 
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len  #Maxmimum length of sentence
        self.dropout = nn.Dropout(dropout)

        #Calculating positional embedding for each word
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # Single dimension tensor ranging from 0 to max_seq_len
        divison_term = torch.exp(torch.arange(0 , d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * divison_term)
        pe[:, 1::2] = torch.cos(position * divison_term)

        pe = pe.unsqueeze(0) # Adds extra dimension to the tensor for batch of sentences

        self.register_buffer('pe', pe) #Will be saved with the model for future use (Encodings dont change)

    def forward(self, x):
        x = x + (self.pe[: , :x.shape[1], : ]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, epsilon : float = 10**-5):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplied for amplication
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
    

class FeedForward(nn.Module):

    def __init__(self, d_model : int, d_ff : int, dropout : int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model , d_ff) # Contains Bias term by default in nn.Linear 
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff , d_model) # Contains Bias term by default in nn.Linear
    
    def forward(self, x):
    # Converts first (Batch , Sequence Lenth, d_model) to (Batch , Sequence Lenth, dff) then again to (Batch , Sequence Lenth, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttentitonBlock(nn.Module):

    def __init__(self, d_model : int, num_heads : int, dropout : int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model is not divisible"
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model , d_model)
        self.w_k = nn.Linear(d_model , d_model)
        self.w_v = nn.Linear(d_model , d_model)
        self.w_o = nn.Linear(d_model , d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2 , -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = attention_scores.softmax(dim = -1) # Batch , h , seq_len , seq_len

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask ):

        query = self.w_q(q) # Query
        key = self.w_k(k) # Key
        value = self.w_v(v) # Value

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2) # Batch, Sequence Length , d_model -> Batch, h, Sequence Length , d_k
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, attention_scores = MultiHeadAttentitonBlock.attention(query, key, value, mask, self.dropout)
        # x is batch , h , sequence length , d_k

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1 , self.num_heads*self.d_k) # becomes Batch, sequence length , d_model

        return self.w_o(x) # Same shape as before
    
class ResidualConnectionLayer(nn.Module):

    def __init__(self, dropout : int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalizaton = LayerNormalization()
    
    def forward(self, x , previous_layer):
        return x + self.dropout(previous_layer(self.normalizaton(x)))
    
class Encoderblock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttentitonBlock, feed_forward_layer : FeedForward, dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_layer = feed_forward_layer
        self.residual_connection_layer = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(2)])

    def forward(self, x , source_mask):
        x = self.residual_connection_layer[0](x, lambda x : self.self_attention_block(x,x, x, source_mask))
        x = self.residual_connection_layer[1](x , self.feed_forward_layer)
        return x

class Encoder(nn.Module):

    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block : MultiHeadAttentitonBlock, cross_attention_block : MultiHeadAttentitonBlock, feed_forward_block : FeedForward, dropout : float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_layer = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, source_mask, target_mask):
        x = self.residual_connection_layer[0](x, lambda x : self.self_attention_block(x , x, x, target_mask))
        x = self.residual_connection_layer[1](x, lambda x : self.cross_attention_block(x , encoder_output, encoder_output, source_mask))
        x = self.residual_connection_layer[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers : nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x , encoder_output, src_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model : int, vocabulary_size : int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocabulary_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim= -1) # returns in Batch size, sequence length , vocabulary size
    
class Trasformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder : Decoder, source_embedding : InputEmbedding, target_embedding : InputEmbedding, source_positions : PositionalEncoding, target_positions : PositionalEncoding, projection_layer : ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.source_positions = source_positions
        self.target_positions = target_positions
        self.projection_layer = projection_layer

    def encode(self, source, source_mask):
        source = self.source_embedding(source)
        source = self.source_positions(source)
        return self.encoder(source, source_mask)
    
    def decode(self, encoder_output, source_mask , target, target_mask):
        target = self.target_embedding(target)
        target = self.target_positions(target)
        return self.decoder(target, encoder_output, source_mask, target_mask)
    
    def projection(self, x):
        return self.projection_layer(x)
    
def build_transformer(source_vocabulary_size : int,  target_vocabulary_size : int, source_sequence_length : int, target_sequence_length : int, d_model : int = 512, N : int = 6, heads : int = 8, dropout : float = 0.1, d_ff : int = 2048) -> None:

    #Create Embeddings for encoder and decoder
    source_embedding = InputEmbedding(d_model, source_vocabulary_size)
    target_embedding = InputEmbedding(d_model, target_vocabulary_size)

    #Create Positional encoding for encoder and decoder
    source_positions = PositionalEncoding(d_model, source_sequence_length, dropout)
    target_positions = PositionalEncoding(d_model, target_sequence_length, dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentitonBlock(d_model, heads, dropout)
        feed_forward_block = FeedForward(d_model , d_ff , dropout)
        encoder_block = Encoderblock(encoder_self_attention,  feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention = MultiHeadAttentitonBlock(d_model, heads, dropout)
        decoder_cross_attention = MultiHeadAttentitonBlock(d_model , heads, dropout)
        feed_forward_block = FeedForward(d_model , d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Defining the Encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating the projection layer for final scores
    projection_layer = ProjectionLayer(d_model, target_vocabulary_size)

    # Creating the Transformer model
    transformer = Trasformer(encoder, decoder, source_embedding, target_embedding, source_positions, target_positions, projection_layer)

    # Initializing the initial parameters

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer