import torch
import torch.nn as nn
import math

class TokenAndPositionEmbedding(nn.Module):
    """
    Combines token and position embeddings.
    """
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_emb(x)
        p = self.pos_emb(positions)
        return x + p

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder block implementation.
    Consists of multi-head self-attention and a position-wise feed-forward network.
    """
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim),
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs, mask=None):
        # PyTorch's MultiheadAttention expects a boolean mask where True indicates a value should be ignored.
        # Keras expects a 0/1 mask. The logic is slightly different.
        # Here we assume a padding mask where `True` is a padded position.
        attention_output, _ = self.attention(
            query=inputs, value=inputs, key=inputs, key_padding_mask=mask
        )
        # Residual connection and layer norm
        proj_input = self.layernorm_1(inputs + attention_output)
        # Feed-forward network
        proj_output = self.dense_proj(proj_input)
        # Residual connection and layer norm
        return self.layernorm_2(proj_input + proj_output)

class BERTModel(nn.Module):
    """
    The main BERT model architecture.
    """
    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoder(embed_dim, ff_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, inputs, padding_mask=None):
        x = self.embedding(inputs)
        for layer in self.encoder_layers:
            x = layer(x, mask=padding_mask)
        return x

def create_bert_model_pytorch(
    vocab_size,
    max_len,
    embed_dim,
    num_heads,
    ff_dim,
    num_layers
):
    """Factory function to create the BERT model."""
    return BERTModel(vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers)


if __name__ == '__main__':
    # Configuration for a small BERT model for demonstration
    VOCAB_SIZE = 20000
    MAX_LEN = 256
    EMBED_DIM = 128
    NUM_HEADS = 4
    FF_DIM = 128
    NUM_LAYERS = 2

    # Create the model
    bert_model = create_bert_model_pytorch(
        VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS
    )
    
    # Print model architecture
    print(bert_model)

    # Example forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, MAX_LEN)).to(device)
    output = bert_model(dummy_input)
    print("Output shape:", output.shape)
