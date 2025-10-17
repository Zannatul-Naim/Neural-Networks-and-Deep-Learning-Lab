import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert_model import create_bert_model_pytorch
import numpy as np
import os
import requests
import tarfile
from collections import Counter
from tqdm import tqdm

# --- Configuration ---
# Model Hyperparameters
VOCAB_SIZE = 20000
MAX_LEN = 256
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 3 # Increase for better performance
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Masking Configuration
MASK_PROBABILITY = 0.15

# File paths
MODEL_WEIGHTS_PATH = "bert_mlm_pretrain_weights_pytorch.pth"
BERT_BACKBONE_WEIGHTS_PATH = "bert_backbone_weights_pytorch.pth"

# --- Data Preparation ---
def download_and_extract_imdb():
    """Downloads and extracts the IMDB dataset."""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target_path = "aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb"):
        print("Downloading IMDB dataset...")
        response = requests.get(url, stream=True)
        with open(target_path, "wb") as f:
            f.write(response.content)
        print("Extracting...")
        with tarfile.open(target_path, "r:gz") as tar:
            tar.extractall()
    
    texts = []
    dataset_dir = "aclImdb"
    for folder in ['train', 'test']:
        for subfolder in ['pos', 'neg']:
            path = os.path.join(dataset_dir, folder, subfolder)
            for fname in os.listdir(path):
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    print(f"Loaded {len(texts)} text documents.")
    return texts

class SimpleTokenizer:
    """A simple tokenizer to build a vocabulary and convert text to IDs."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
        
    def adapt(self, texts):
        print("Adapting tokenizer...")
        words = " ".join(texts).split()
        word_counts = Counter(words)
        
        # Start vocab with special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            
        # Add most common words
        for i, (word, _) in enumerate(word_counts.most_common(self.vocab_size - len(self.special_tokens))):
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
            
    def encode(self, texts, max_len):
        encoded = []
        for text in texts:
            tokens = text.split()
            ids = [self.word_to_id.get(token, self.special_tokens["[UNK]"]) for token in tokens]
            ids = ids[:max_len]
            # Pad
            ids += [self.special_tokens["[PAD]"]] * (max_len - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)
    
    def get_vocab_size(self):
        return len(self.word_to_id)
    
    def get_mask_token_id(self):
        return self.special_tokens["[MASK]"]

class MLMDataset(Dataset):
    """PyTorch Dataset for Masked Language Modeling."""
    def __init__(self, vectorized_texts, tokenizer):
        self.vectorized_texts = vectorized_texts
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.get_mask_token_id()
        self.vocab_size = tokenizer.get_vocab_size()

    def __len__(self):
        return len(self.vectorized_texts)

    def __getitem__(self, idx):
        # Start with a copy
        input_ids = self.vectorized_texts[idx].clone()
        labels = torch.full_like(input_ids, -100) # -100 is the ignore_index in CrossEntropyLoss

        # Create a mask for tokens to potentially mask
        can_be_masked = (input_ids != 0) & (input_ids != 1) # Not PAD or UNK
        mask_rand = torch.rand(input_ids.shape)
        
        # Select 15% of the tokens
        mask = mask_rand < MASK_PROBABILITY
        mask &= can_be_masked
        
        labels[mask] = input_ids[mask]

        # 80% of masked tokens become [MASK]
        mask_to_mask_token = torch.rand(input_ids.shape) < 0.8
        input_ids[mask & mask_to_mask_token] = self.mask_token_id

        # 10% become a random token
        mask_to_random_token = torch.rand(input_ids.shape) < 0.5 # 0.5 of the remaining 0.2
        random_words = torch.randint(3, self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[mask & ~mask_to_mask_token & mask_to_random_token] = random_words[mask & ~mask_to_mask_token & mask_to_random_token]

        return {"input_ids": input_ids, "labels": labels}

class MaskedLanguageModel(nn.Module):
    """Wrapper model for the MLM task. Combines BERT with a prediction head."""
    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.mlm_head = nn.Linear(bert.encoder_layers[0].attention.embed_dim, vocab_size)

    def forward(self, input_ids, padding_mask=None):
        bert_output = self.bert(input_ids, padding_mask=padding_mask)
        return self.mlm_head(bert_output)

def main():
    """Main pre-training script."""
    # 1. Prepare Data
    texts = download_and_extract_imdb()
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    tokenizer.adapt(texts)
    
    # Ensure vocab size is consistent
    actual_vocab_size = tokenizer.get_vocab_size()

    all_vectorized_data = tokenizer.encode(texts, MAX_LEN)

    # 2. Create Dataset and DataLoader
    mlm_dataset = MLMDataset(all_vectorized_data, tokenizer)
    mlm_dataloader = DataLoader(mlm_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Create BERT and MLM models
    bert_backbone = create_bert_model_pytorch(
        actual_vocab_size, MAX_LEN, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS
    )
    mlm_model = MaskedLanguageModel(bert_backbone, actual_vocab_size).to(DEVICE)
    
    # 4. Set up optimizer and loss function
    optimizer = torch.optim.Adam(mlm_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 5. Training loop
    print("Starting MLM pre-training...")
    for epoch in range(EPOCHS):
        mlm_model.train()
        total_loss = 0
        
        progress_bar = tqdm(mlm_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            
            padding_mask = (input_ids == 0)
            outputs = mlm_model(input_ids, padding_mask=padding_mask)
            # outputs = mlm_model(input_ids)
            
            # Reshape for loss calculation
            loss = criterion(outputs.view(-1, actual_vocab_size), labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(mlm_dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # 6. Save weights
    print(f"Saving full MLM model state dict to {MODEL_WEIGHTS_PATH}")
    torch.save(mlm_model.state_dict(), MODEL_WEIGHTS_PATH)
    
    print(f"Saving BERT backbone state dict to {BERT_BACKBONE_WEIGHTS_PATH}")
    torch.save(bert_backbone.state_dict(), BERT_BACKBONE_WEIGHTS_PATH)
    print("Pre-training complete.")

if __name__ == "__main__":
    main()
