import argparse
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import pickle
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Union

# Import our model implementations
# Assuming the models are defined in a file called models.py
from models import RNNLanguageModel, LSTMLanguageModel, TransformerLanguageModel

# Configure logging
def setup_logging(log_dir, model_type):
    """Set up logging to file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_type}_training_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Dataset class for handling tokenized sequences
class TokenizedDataset(Dataset):
    def __init__(self, data_path, context_length=128, logger=None):
        """
        Dataset for training language models on tokenized sequences.
        
        Args:
            data_path: Path to the tokenized data file (.pkl or .npz)
            context_length: Maximum context length for training
            logger: Logger instance
        """
        self.context_length = context_length
        self.logger = logger or logging.getLogger('training')
        
        # Load tokenized data
        self.logger.info(f"Loading tokenized data from {data_path}")
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.sequences = pickle.load(f)
        elif data_path.endswith('.npz'):
            data = np.load(data_path)
            sequences = data['sequences']
            lengths = data['lengths']
            
            # Reconstruct original sequences
            self.sequences = []
            for i, length in enumerate(lengths):
                self.sequences.append(sequences[i, :length])
        else:
            error_msg = f"Unsupported file format: {data_path}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Loaded {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Gets a sequence for training.
        
        For each sequence, we:
        1. Truncate to maximum context length + 1
        2. Use first N tokens as input
        3. Use tokens 1 to N+1 as targets (shifted by 1)
        """
        # Get the tokenized sequence
        sequence = self.sequences[idx]
        
        # Truncate to context_length + 1 (need one extra for targets)
        if len(sequence) > self.context_length + 1:
            start_idx = random.randint(0, len(sequence) - self.context_length - 1)
            sequence = sequence[start_idx:start_idx + self.context_length + 1]
        
        # Convert to tensors
        tokens = torch.tensor(sequence[:-1], dtype=torch.long)
        targets = torch.tensor(sequence[1:], dtype=torch.long)
        
        return tokens, targets

# Collate function for padding batches
def collate_fn(batch):
    """Pad sequences in a batch to same length."""
    inputs, targets = zip(*batch)
    
    # Pad inputs and targets
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded

# Training function
def train_epoch(model, dataloader, optimizer, criterion, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size, seq_len = inputs.size()
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(inputs)
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        # Accumulate loss stats
        total_loss += loss.item() * seq_len * batch_size
        total_tokens += seq_len * batch_size
        
        # Progress reporting
        if (batch_idx + 1) % 50 == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / 50
            logger.info(f"Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | {ms_per_batch:.2f} ms/batch")
            start_time = time.time()
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

# Evaluation function
def evaluate(model, dataloader, criterion, device, logger):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size, seq_len = inputs.size()
            
            # Forward pass
            logits = model(inputs)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            # Accumulate loss stats
            total_loss += loss.item() * seq_len * batch_size
            total_tokens += seq_len * batch_size
    
    # Calculate average loss
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

# Generate sample text based on prompts
def generate_samples(model, tokenizer, prompts=None, temperature=0.8, max_length=50, logger=None):
    """Generate sample text from the model."""
    logger = logger or logging.getLogger('training')
    model.eval()
    
    # Default prompts if none are provided
    if prompts is None or len(prompts) == 0:
        prompts = [
            "Once upon a time",
            "In a distant galaxy",
            "The meaning of life"
        ]
    
    logger.info("\nGenerating sample text:")
    logger.info("=" * 40)
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Prompt {i+1}: {prompt}")
        
        # Generate text
        generated = model.prompt(prompt, max_seq_length=max_length, temperature=temperature)
        logger.info(f"Generated: {generated}")
        logger.info("-" * 40)

# Main training function
def train_model(args, logger):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Load the tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(args.tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = TokenizedDataset(args.train_data, context_length=args.context_length, logger=logger)
    valid_dataset = TokenizedDataset(args.valid_data, context_length=args.context_length, logger=logger) if args.valid_data else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    ) if valid_dataset else None
    
    # Initialize model
    logger.info(f"Initializing {args.model} model...")
    
    if args.model == 'rnn':
        model = RNNLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tokenizer_path=args.tokenizer_path
        )
    elif args.model == 'lstm':
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            tokenizer_path=args.tokenizer_path
        )
    elif args.model == 'transformer':
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            nhead=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_length=512,
            tokenizer_path=args.tokenizer_path
        )
    else:
        error_msg = f"Unsupported model type: {args.model}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    model = model.to(device)
    
    # Log model summary
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=False  # Disable verbose output since we're logging manually
    )
    
    # Parse custom prompts if provided
    custom_prompts = []
    if args.prompts:
        custom_prompts = args.prompts
        logger.info(f"Using {len(custom_prompts)} custom prompts for text generation")
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_count = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        logger.info("-" * 20)
        
        # Train
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        logger.info(f"Train Loss: {train_loss:.4f} | Train Perplexity: {train_ppl:.2f}")
        
        # Validate
        if valid_loader:
            val_loss, val_ppl = evaluate(model, valid_loader, criterion, device, logger)
            logger.info(f"Valid Loss: {val_loss:.4f} | Valid Perplexity: {val_ppl:.2f}")
            
            # Update learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            if old_lr != new_lr:
                logger.info(f"Learning rate changed from {old_lr} to {new_lr}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                early_stop_count = 0
                
                # Save best model
                save_path = os.path.join(args.output_dir, f"{args.model}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'args': vars(args)
                }, save_path)
                logger.info(f"Model saved to {save_path}")
                
                # Generate samples with best model
                generate_samples(model, tokenizer, custom_prompts, 
                               temperature=args.temperature, max_length=args.gen_max_length, 
                               logger=logger)
            else:
                early_stop_count += 1
                logger.info(f"No improvement. Early stopping count: {early_stop_count}/{args.patience}")
                
            # Early stopping
            if early_stop_count >= args.patience:
                logger.info(f"Early stopping after {epoch} epochs.")
                break
    
    # Save final model
    final_path = os.path.join(args.output_dir, f"{args.model}_final.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args)
    }, final_path)
    logger.info(f"Final model saved to {final_path}")
    
    logger.info("\nTraining completed!")
    if valid_loader:
        logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    
    # Load the best model for final sample generation
    if valid_loader and os.path.exists(os.path.join(args.output_dir, f"{args.model}_best.pt")):
        checkpoint = torch.load(os.path.join(args.output_dir, f"{args.model}_best.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("\nGenerating samples with best model:")
        generate_samples(model, tokenizer, custom_prompts, 
                        temperature=args.temperature, max_length=args.gen_max_length, 
                        logger=logger)

def inference_only(args, logger):
    """Run inference using a pre-trained model."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seeds for reproducibility
    set_seed(args.seed)
    
    # Check if model exists
    model_path = args.model_path
    if not os.path.exists(model_path):
        error_msg = f"Model file not found: {model_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load the checkpoint
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint.get('args', {})
    
    # Load the tokenizer
    tokenizer_path = args.tokenizer_path
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.get_piece_size()
    
    # Determine model type from the file name or args
    model_type = None
    if 'model' in model_args:
        model_type = model_args['model']
    else:
        # Try to infer from filename
        if 'rnn' in model_path.lower():
            model_type = 'rnn'
        elif 'lstm' in model_path.lower():
            model_type = 'lstm'
        elif 'transformer' in model_path.lower():
            model_type = 'transformer'
        else:
            error_msg = "Could not determine model type from model path"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    logger.info(f"Creating {model_type} model instance")
    
    # Initialize model based on type
    if model_type == 'rnn':
        model = RNNLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=model_args.get('embedding_dim', args.embedding_dim),
            hidden_dim=model_args.get('hidden_dim', args.hidden_dim),
            num_layers=model_args.get('num_layers', args.num_layers),
            dropout=model_args.get('dropout', args.dropout),
            tokenizer_path=tokenizer_path
        )
    elif model_type == 'lstm':
        model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=model_args.get('embedding_dim', args.embedding_dim),
            hidden_dim=model_args.get('hidden_dim', args.hidden_dim),
            num_layers=model_args.get('num_layers', args.num_layers),
            dropout=model_args.get('dropout', args.dropout),
            tokenizer_path=tokenizer_path
        )
    elif model_type == 'transformer':
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=model_args.get('embedding_dim', args.embedding_dim),
            hidden_dim=model_args.get('hidden_dim', args.hidden_dim),
            nhead=model_args.get('num_heads', args.num_heads),
            num_layers=model_args.get('num_layers', args.num_layers),
            dropout=model_args.get('dropout', args.dropout),
            max_seq_length=512,
            tokenizer_path=tokenizer_path
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # Parse custom prompts
    custom_prompts = args.prompts if args.prompts else []
    if not custom_prompts:
        logger.info("No prompts provided. Using default prompts.")
        custom_prompts = [
            "Once upon a time",
            "In a distant galaxy",
            "The meaning of life"
        ]
    
    # Generate text
    logger.info(f"Generating text with temperature {args.temperature}")
    generate_samples(model, tokenizer, custom_prompts, 
                   temperature=args.temperature, max_length=args.gen_max_length, 
                   logger=logger)

def main():
    parser = argparse.ArgumentParser(description="Train language models on tokenized text or run inference")
    
    # Mode selection
    parser.add_argument("--inference_only", action="store_true",
                        help="Run only inference without training")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, 
                        help="Path to training data file (.pkl or .npz)")
    parser.add_argument("--valid_data", type=str, default=None, 
                        help="Path to validation data file (.pkl or .npz)")
    parser.add_argument("--tokenizer_path", type=str, required=True, 
                        help="Path to the SentencePiece tokenizer model")
    
    # Model arguments
    parser.add_argument("--model", type=str, choices=["rnn", "lstm", "transformer"],
                        help="Type of model to train")
    parser.add_argument("--model_path", type=str, 
                        help="Path to pre-trained model checkpoint for inference")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Token embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers in the model")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads (transformer only)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout probability")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Training batch size")
    parser.add_argument("--context_length", type=int, default=128,
                        help="Context length for training")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay factor")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    
    # Generation arguments
    parser.add_argument("--prompts", type=str, nargs="+", default=[],
                        help="Prompts for text generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for text generation (higher = more random)")
    parser.add_argument("--gen_max_length", type=int, default=100,
                        help="Maximum generation length")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save training logs")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.inference_only:
        if not args.model_path:
            raise ValueError("--model_path is required for inference mode")
    else:
        if not args.train_data:
            raise ValueError("--train_data is required for training mode")
        if not args.model:
            raise ValueError("--model is required for training mode")
    
    # Create output directory if it doesn't exist
    if not args.inference_only:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    model_name = args.model or "inference"
    logger = setup_logging(args.log_dir, model_name)
    
    # Log all arguments
    logger.info(f"Running in {'inference' if args.inference_only else 'training'} mode")
    logger.info("Parameters:")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    
    try:
        if args.inference_only:
            # Run inference
            inference_only(args, logger)
        else:
            # Train the model
            train_model(args, logger)
    except Exception as e:
        logger.exception(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
