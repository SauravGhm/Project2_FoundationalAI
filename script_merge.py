import os
import random
import argparse
from pathlib import Path
import re

def sentence_level_split(root_folder, train_file="train.txt", test_file="test.txt", train_ratio=0.8, seed=42):
    """
    Read all text files in a directory and split at the sentence level to create
    train.txt and test.txt files.
    
    Args:
        root_folder (str): Path to the folder containing text files
        train_file (str): Name of the output training file
        test_file (str): Name of the output testing file
        train_ratio (float): Ratio of sentences to be used for training (0.0-1.0)
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Get all text files in the root folder
    txt_files = list(Path(root_folder).glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {root_folder}")
        return
    
    print(f"Found {len(txt_files)} text files in {root_folder}")
    
    # Collect all sentences
    all_sentences = []
    
    for i, file_path in enumerate(txt_files):
        if i % 10 == 0 or i == len(txt_files) - 1:
            print(f"Reading file {i+1}/{len(txt_files)}: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
                # More sophisticated sentence splitting
                # Split on periods, exclamation marks, and question marks followed by whitespace or end of string
                sentences = re.split(r'(?<=[.!?])\s+', content)
                
                # Clean up sentences
                clean_sentences = []
                for s in sentences:
                    s = s.strip()
                    if s:  # Only keep non-empty sentences
                        if not s.endswith(('.', '!', '?')):
                            s += '.'
                        clean_sentences.append(s)
                
                all_sentences.extend(clean_sentences)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Report the number of sentences found
    print(f"\nExtracted {len(all_sentences)} sentences from all files")
    
    # Shuffle and split sentences
    random.shuffle(all_sentences)
    split_idx = int(len(all_sentences) * train_ratio)
    train_sentences = all_sentences[:split_idx]
    test_sentences = all_sentences[split_idx:]
    
    # Write to output files
    print(f"Writing {len(train_sentences)} sentences to training file...")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_sentences))
    
    print(f"Writing {len(test_sentences)} sentences to testing file...")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_sentences))
    
    # Display information about the created files
    train_size = os.path.getsize(train_file) / (1024 * 1024)
    test_size = os.path.getsize(test_file) / (1024 * 1024)
    
    print(f"\nSplit complete:")
    print(f"- Training set: {len(train_sentences)} sentences, {train_size:.2f} MB ({train_ratio*100:.0f}%)")
    print(f"- Testing set: {len(test_sentences)} sentences, {test_size:.2f} MB ({(1-train_ratio)*100:.0f}%)")
    print(f"\nFiles created successfully: {train_file} and {test_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split text files at sentence level into train and test sets")
    parser.add_argument("root_folder", help="Folder containing the text files")
    parser.add_argument("--train_file", default="train.txt", help="Output training file name")
    parser.add_argument("--test_file", default="test.txt", help="Output testing file name")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of sentences for training (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    sentence_level_split(
        args.root_folder, 
        args.train_file, 
        args.test_file, 
        args.train_ratio, 
        args.seed
    )
