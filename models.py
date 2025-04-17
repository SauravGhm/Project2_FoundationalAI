import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
import numpy as np
from typing import List, Union, Optional, Tuple

import tokenize
tokenize.open = open
class BaseLanguageModel(nn.Module):
    """Base class for all language models with common functionality."""
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 tokenizer_path: str,
                 padding_idx: int = 0):
        """
        Initialize the base language model.
        """
        super(BaseLanguageModel, self).__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Output layer (maps hidden states to logits)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Constants
        self.eos_id = 3  # Assuming </s> has ID 3, modify if different
        self.vocab_size = vocab_size
        
    def sample_token(self, logits: torch.Tensor, temperature: float = 1.0) -> int:
        """
        Sample a token from the output probability distribution.
        """
        if temperature == 0:
            # Greedy sampling (no randomness)
            return torch.argmax(logits, dim=-1).item()
        else:
            # Apply temperature to logits
            scaled_logits = logits / temperature
            
            # Convert to probabilities
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Sample from the distribution
            return torch.multinomial(probs, 1).item()
    
    def prompt(self, 
              text: str, 
              max_seq_length: int = 100, 
              temperature: float = 1.0) -> str:
        """
        Generate text given a prompt.
        """
        # Tokenize the prompt
        tokens = self.tokenizer.encode(text)
        
        # Prepare for generation
        generated_tokens = []
        input_seq = torch.tensor([tokens], dtype=torch.long)
        
        # Move to GPU if available
        device = next(self.parameters()).device
        input_seq = input_seq.to(device)
        
        # Generate tokens auto-regressively
        for _ in range(max_seq_length):
            with torch.no_grad():
                # Get logits for the next token
                logits = self.forward(input_seq)
                
                # Sample the next token (using the last token's prediction)
                next_token = self.sample_token(logits[0, -1], temperature)
                
                # Stop if end of sequence token
                if next_token == self.eos_id:
                    break
                
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Update input sequence for next iteration
                new_token = torch.tensor([[next_token]], dtype=torch.long).to(device)
                input_seq = torch.cat([input_seq, new_token], dim=1)
        
        # Decode the full generated sequence
        response_tokens = tokens + generated_tokens
        response_text = self.tokenizer.decode(response_tokens)
        
        return response_text


class RNNLanguageModel(BaseLanguageModel):
    """RNN-based language model."""
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 tokenizer_path: str = "gutenberg_bpe.model"):
        """
        Initialize the RNN language model.
        """
        super(RNNLanguageModel, self).__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            tokenizer_path=tokenizer_path
        )
        
        # RNN layer(s)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Save hyperparameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, 
               input_ids: torch.Tensor, 
               hidden: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Create embeddings from token IDs
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = input_ids.size(0)
            hidden = torch.zeros(
                self.num_layers, 
                batch_size, 
                self.hidden_dim, 
                device=input_ids.device
            )
        
        # Pass through RNN
        output, hidden = self.rnn(embeddings, hidden)  # output: (batch_size, seq_len, hidden_dim)
        
        # Pass through output layer to get logits
        logits = self.output_layer(output)  # (batch_size, seq_len, vocab_size)
        
        return logits


class LSTMLanguageModel(BaseLanguageModel):
    """LSTM-based language model."""
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.2,
                 tokenizer_path: str = "gutenberg_bpe.model"):
        """
        Initialize the LSTM language model.
        """
        super(LSTMLanguageModel, self).__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            tokenizer_path=tokenizer_path
        )
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Save hyperparameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, 
               input_ids: torch.Tensor, 
               hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:

        # Create embeddings from token IDs
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = input_ids.size(0)
            h0 = torch.zeros(
                self.num_layers, 
                batch_size, 
                self.hidden_dim, 
                device=input_ids.device
            )
            c0 = torch.zeros(
                self.num_layers, 
                batch_size, 
                self.hidden_dim, 
                device=input_ids.device
            )
            hidden = (h0, c0)
        
        # Pass through LSTM
        output, hidden = self.lstm(embeddings, hidden)  # output: (batch_size, seq_len, hidden_dim)
        
        # Pass through output layer to get logits
        logits = self.output_layer(output)  # (batch_size, seq_len, vocab_size)
        
        return logits


class TransformerLanguageModel(BaseLanguageModel):
    """Transformer-based language model."""
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dim: int,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 max_seq_length: int = 512,
                 tokenizer_path: str = "gutenberg_bpe.model"):
    
        super(TransformerLanguageModel, self).__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim, 
            hidden_dim=embedding_dim,  # For transformer, hidden_dim equals embedding_dim
            tokenizer_path=tokenizer_path
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            dropout=dropout,
            max_len=max_seq_length
        )
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_layers
        )
        
        # Update output layer for transformer
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Save parameters
        self.max_seq_length = max_seq_length
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Create embeddings from token IDs
        embeddings = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        
        # Apply positional encoding
        embeddings = self.pos_encoder(embeddings)
        
        # Create mask to prevent attention to future tokens (causal mask)
        # This ensures the model can only attend to previous positions
        seq_len = input_ids.size(1)
        mask = generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        # Apply transformer encoder with mask
        output = self.transformer_encoder(embeddings, mask)
        
        # Apply output layer
        logits = self.output_layer(output)
        
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin and cos functions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
