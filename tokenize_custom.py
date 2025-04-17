import os
import numpy as np
import sentencepiece as spm
from pathlib import Path

# Define constants
VOCAB_SIZE = 10000
MODEL_PREFIX = "gutenberg_bpe"
INPUT_FILE = "train.txt"  # Assuming this is your training dataset
TEST_FILE = "test.txt"    # Assuming this is your test dataset

def prepare_data_for_training():
    """
    If needed, combine all text files into a single file for SentencePiece training.
    This is optional if your data is already in a single file.
    """
    print("Preparing data for tokenizer training...")
    if not os.path.exists(INPUT_FILE):
        print(f"Warning: {INPUT_FILE} not found. Please ensure your training data exists.")
    else:
        print(f"Using existing file {INPUT_FILE} for training.")

def train_tokenizer():
    """
    Train a SentencePiece BPE tokenizer on the input data with vocab size 10000.
    """
    print(f"Training BPE tokenizer with vocabulary size {VOCAB_SIZE}...")
    
    # Set up training parameters
    spm.SentencePieceTrainer.train(
        input=INPUT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",  # Use BPE (Byte Pair Encoding)
        character_coverage=1.0,  # Covers all characters in the training data
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        normalization_rule_name="identity"  # No normalization
    )
    
    print(f"Tokenizer trained and saved as {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")

def load_tokenizer():
    """
    Load the trained SentencePiece tokenizer.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")
    return sp

def tokenize_dataset(tokenizer, file_path, save_path=None):

    print(f"Tokenizing dataset: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return []
    
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Tokenize each line
    tokenized_sequences = []
    for line in lines:
        if line.strip():  # Skip empty lines
            # Get token IDs
            token_ids = tokenizer.encode(line.strip())
            tokenized_sequences.append(token_ids)
    
    print(f"Tokenized {len(tokenized_sequences)} sequences")
    
    # Save the tokenized sequences properly handling variable lengths
    if save_path:
        # Method 1: Save as a list of arrays using pickle (recommended)
        import pickle
        with open(save_path.replace('.npy', '.pkl'), 'wb') as f:
            pickle.dump(tokenized_sequences, f)
        print(f"Saved tokenized sequences to {save_path.replace('.npy', '.pkl')}")
        
        # Method 2: Alternatively, pad sequences to the same length and save as numpy array
        # Find maximum sequence length
        max_length = max(len(seq) for seq in tokenized_sequences)
        print(f"Maximum sequence length: {max_length}")
        
        # Create a padded array
        padded_sequences = np.zeros((len(tokenized_sequences), max_length), dtype=np.int32)
        sequence_lengths = np.zeros(len(tokenized_sequences), dtype=np.int32)
        
        # Fill the padded array and record original lengths
        for i, seq in enumerate(tokenized_sequences):
            length = len(seq)
            sequence_lengths[i] = length
            padded_sequences[i, :length] = seq
        
        # Save both the padded sequences and their original lengths
        np.savez(save_path.replace('.npy', '.npz'), 
                 sequences=padded_sequences, 
                 lengths=sequence_lengths)
        print(f"Saved padded sequences to {save_path.replace('.npy', '.npz')}")
    
    return tokenized_sequences

def analyze_tokenized_data(tokenized_sequences, tokenizer, num_examples=5):
    """
    Analyze tokenized data and show some examples.
    """
    if not tokenized_sequences:
        return
    
    print("\nAnalysis of tokenized data:")
    print(f"Number of sequences: {len(tokenized_sequences)}")
    
    # Length statistics
    lengths = [len(seq) for seq in tokenized_sequences]
    print(f"Average sequence length: {np.mean(lengths):.2f} tokens")
    print(f"Min sequence length: {min(lengths)} tokens")
    print(f"Max sequence length: {max(lengths)} tokens")
    
    # Show some examples
    print(f"\nShowing {num_examples} tokenization examples:")
    
    for i in range(min(num_examples, len(tokenized_sequences))):
        # Get a random sequence if there are many
        if len(tokenized_sequences) > num_examples:
            idx = np.random.randint(0, len(tokenized_sequences))
        else:
            idx = i
            
        token_ids = tokenized_sequences[idx]
        # Convert token IDs back to pieces
        tokens = tokenizer.id_to_piece(token_ids)
        
        # Display the tokens and their IDs
        print(f"\nExample {i+1}:")
        print(f"Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        
        # Show reconstructed text
        decoded_text = tokenizer.decode(token_ids)
        print(f"Decoded text: {decoded_text[:100]}{'...' if len(decoded_text) > 100 else ''}")

def load_tokenized_dataset(file_path):
    
    # Check file extension
    if file_path.endswith('.pkl'):
        # Load pickle file
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    elif file_path.endswith('.npz'):
        # Load npz file with padded sequences and their lengths
        data = np.load(file_path)
        sequences = data['sequences']
        lengths = data['lengths']
        
        # Reconstruct the original sequences
        tokenized_sequences = []
        for i, length in enumerate(lengths):
            tokenized_sequences.append(sequences[i, :length])
        
        return tokenized_sequences
    
    else:
        print(f"Unsupported file format: {file_path}")
        return []

def main():
    """
    Main function to execute the tokenization pipeline.
    """
    print("Starting BPE tokenization pipeline with SentencePiece")
    
    # Step 1: Prepare data (if needed)
    prepare_data_for_training()
    
    # Step 2: Train the tokenizer if model doesn't already exist
    if not os.path.exists(f"{MODEL_PREFIX}.model"):
        train_tokenizer()
    else:
        print(f"Using existing tokenizer model: {MODEL_PREFIX}.model")
    
    # Step 3: Load the trained tokenizer
    tokenizer = load_tokenizer()
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.get_piece_size()}")
    
    # Step 4: Tokenize training and test datasets
    train_tokenized = tokenize_dataset(tokenizer, INPUT_FILE, save_path="train_tokenized.npy")
    test_tokenized = tokenize_dataset(tokenizer, TEST_FILE, save_path="test_tokenized.npy")
    
    # Step 5: Analyze the tokenized data
    analyze_tokenized_data(train_tokenized, tokenizer)
    
    print("\nTokenization process completed!")
    print("Tokenized datasets saved in both pickle (.pkl) and numpy (.npz) formats")
    print("Use the pickle format for variable length sequences")
    print("Use the numpy format for padded sequences with recorded lengths")

if __name__ == "__main__":
    main()

