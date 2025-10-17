import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from bert_model import create_bert_model_pytorch
import numpy as np
import os
import json
import tarfile
import requests
import zipfile
import string
import re
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset

# --- Global Configuration ---
VOCAB_SIZE = 20000
MAX_LEN_CLASSIFICATION = 256
MAX_LEN_QA = 384
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_WEIGHTS_PATH = "bert_backbone_weights_pytorch.pth"
LEARNING_RATE = 3e-5

# --- Reusable Tokenizer ---
class SimpleTokenizer:
    """A robust tokenizer shared across all tasks."""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2, "[SEP]": 3}

    def _tokenize_text(self, text):
        text = text.lower()
        text = re.sub(r"([?.!,¿\"])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        return text.strip().split(' ')

    def adapt(self, texts):
        print("Adapting tokenizer on IMDB corpus...")
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            all_tokens.extend(self._tokenize_text(text))
        
        word_counts = Counter(all_tokens)
        
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
            
        for i, (word, _) in enumerate(word_counts.most_common(self.vocab_size - len(self.special_tokens))):
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
    
    def encode(self, texts, max_len):
        encoded = []
        unk_id = self.special_tokens["[UNK]"]
        pad_id = self.special_tokens["[PAD]"]
        for text in texts:
            tokens = self._tokenize_text(text)
            ids = [self.word_to_id.get(t, unk_id) for t in tokens]
            ids = ids[:max_len]
            ids += [pad_id] * (max_len - len(ids))
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def encode_qa(self, questions, contexts, max_len):
        encoded = []
        sep_id = self.special_tokens["[SEP]"]
        pad_id = self.special_tokens["[PAD]"]
        unk_id = self.special_tokens["[UNK]"]
        for q, c in zip(questions, contexts):
            q_ids = [self.word_to_id.get(t, unk_id) for t in self._tokenize_text(q)]
            c_ids = [self.word_to_id.get(t, unk_id) for t in self._tokenize_text(c)]
            combined = q_ids + [sep_id] + c_ids
            combined = combined[:max_len]
            combined += [pad_id] * (max_len - len(combined))
            encoded.append(combined)
        return torch.tensor(encoded, dtype=torch.long)

    def decode(self, token_ids):
        tokens = [self.id_to_word.get(idx, '[UNK]') for idx in token_ids if idx != self.special_tokens['[PAD]']]
        return " ".join(tokens).replace(" ?", "?").replace(" .", ".").replace(" !", "!").replace(" ,", ",")
        
    def get_vocab_size(self): return len(self.word_to_id)

def get_tokenizer():
    """Builds and returns a tokenizer based on the IMDB dataset."""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb"):
        print("Downloading IMDB for tokenizer vocabulary...")
        response = requests.get(url, stream=True)
        with open("aclImdb_v1.tar.gz", "wb") as f: f.write(response.content)
        with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar: tar.extractall()
    texts = []
    for folder in ['train', 'test']:
        for sub in ['pos', 'neg']:
            path = os.path.join("aclImdb", folder, sub)
            for fname in os.listdir(path):
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    tokenizer.adapt(texts)
    return tokenizer

# --- Task 1: Sentiment Classification ---
def finetune_sentiment_classification(tokenizer):
    print("\n" + "="*60)
    print("Task 1: Fine-Tuning for Sentiment Classification (IMDB)")
    print("="*60)

    class SentimentModel(nn.Module):
        def __init__(self, bert, num_classes=1):
            super().__init__()
            self.bert = bert
            self.classifier = nn.Linear(EMBED_DIM, num_classes)
        def forward(self, input_ids):
            bert_output = self.bert(input_ids)
            cls_output = bert_output[:, 0, :]
            return self.classifier(cls_output)

    class ImdbDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.encoded = tokenizer.encode(texts, MAX_LEN_CLASSIFICATION)
            self.labels = torch.tensor(labels, dtype=torch.float)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return {"input_ids": self.encoded[idx], "label": self.labels[idx]}

    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for split, texts, labels in [('train', train_texts, train_labels), ('test', test_texts, test_labels)]:
        for label_val, sentiment in enumerate(['neg', 'pos']):
            path = os.path.join('aclImdb', split, sentiment)
            for fname in os.listdir(os.path.join(path))[:5000]: # Using a subset
                with open(os.path.join(path, fname), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(label_val)

    train_dataset = ImdbDataset(train_texts, train_labels, tokenizer)
    test_dataset = ImdbDataset(test_texts, test_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    bert = create_bert_model_pytorch(tokenizer.get_vocab_size(), MAX_LEN_CLASSIFICATION, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        bert.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE), strict=False)
        print("Loaded pre-trained weights for sentiment model.")
    model = SentimentModel(bert).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    model.train()
    for epoch in range(1): # Reduced epochs for speed
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Sentiment"):
            input_ids, labels = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
            outputs = model(input_ids).squeeze(1)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f"Sentiment Classification Accuracy: {acc:.4f}")
    return acc

# --- Task 2: Semantic Similarity (SNLI) ---
def finetune_semantic_similarity(tokenizer):
    print("\n" + "="*60)
    print("Task 2: Fine-Tuning for Semantic Similarity (SNLI)")
    print("="*60)

    class SimilarityModel(nn.Module):
        def __init__(self, bert, num_classes=3):
            super().__init__()
            self.bert = bert
            self.classifier = nn.Linear(EMBED_DIM, num_classes)
        def forward(self, input_ids):
            return self.classifier(self.bert(input_ids)[:, 0, :])

    class SnliDataset(Dataset):
        def __init__(self, premises, hypotheses, labels, tokenizer):
            texts = [p + " [SEP] " + h for p, h in zip(premises, hypotheses)]
            self.encoded = tokenizer.encode(texts, MAX_LEN_CLASSIFICATION)
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return {"input_ids": self.encoded[idx], "label": self.labels[idx]}

    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    if not os.path.exists("snli_1.0"):
        print("Downloading SNLI dataset...")
        r = requests.get(url)
        with open("snli_1.0.zip", 'wb') as f: f.write(r.content)
        with zipfile.ZipFile("snli_1.0.zip", 'r') as z: z.extractall(".")
    
    def read_snli(fp):
        p, h, l = [], [], []
        label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}
        with open(fp, 'r') as f:
            for line in f:
                d = json.loads(line)
                if d['gold_label'] in label_map:
                    p.append(d['sentence1'])
                    h.append(d['sentence2'])
                    l.append(label_map[d['gold_label']])
        return p, h, l

    train_p, train_h, train_l = read_snli("snli_1.0/snli_1.0_train.jsonl")
    val_p, val_h, val_l = read_snli("snli_1.0/snli_1.0_dev.jsonl")

    train_ds = SnliDataset(train_p[:10000], train_h[:10000], train_l[:10000], tokenizer)
    val_ds = SnliDataset(val_p[:1000], val_h[:1000], val_l[:1000], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    bert = create_bert_model_pytorch(tokenizer.get_vocab_size(), MAX_LEN_CLASSIFICATION, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
    if os.path.exists(PRETRAINED_WEIGHTS_PATH): 
        bert.load_state_dict(torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE), strict=False)
        print("Loaded pre-trained weights for similarity model.")
    model = SimilarityModel(bert).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1): # Reduced epochs for speed
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} SNLI"):
            input_ids, labels = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch['input_ids'].to(DEVICE), batch['label'].to(DEVICE)
            _, predicted = torch.max(model(input_ids).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f"SNLI Accuracy: {acc:.4f}")
    return acc

# --- Task 3: Question Answering (SQuAD) ---
def finetune_question_answering(tokenizer):
    print("\n" + "="*60)
    print("Task 3: Fine-Tuning for Question Answering (SQuAD)")
    print("="*60)
    
    class BERTForQuestionAnswering(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
            self.qa_outputs = nn.Linear(EMBED_DIM, 2)
        def forward(self, input_ids):
            logits = self.qa_outputs(self.bert(input_ids))
            start_logits, end_logits = logits.split(1, dim=-1)
            return start_logits.squeeze(-1), end_logits.squeeze(-1)

    def find_answer_positions(questions, contexts, answers, tokenizer, max_len):
        start_pos, end_pos = [], []
        for q, c, a_dict in zip(questions, contexts, answers):
            q_tok = tokenizer._tokenize_text(q)
            c_tok = tokenizer._tokenize_text(c)
            offset = len(q_tok) + 1
            ans_text = a_dict['text'][0] if a_dict['text'] else ""
            ans_tok = tokenizer._tokenize_text(ans_text)
            s, e = 0, 0
            for i in range(len(c_tok) - len(ans_tok) + 1):
                if c_tok[i:i+len(ans_tok)] == ans_tok:
                    s = offset + i
                    e = offset + i + len(ans_tok) - 1
                    break
            start_pos.append(min(s, max_len - 1))
            end_pos.append(min(e, max_len - 1))
        return torch.tensor(start_pos, dtype=torch.long), torch.tensor(end_pos, dtype=torch.long)

    class SQuADDataset(Dataset):
        def __init__(self, q, c, a, tokenizer):
            self.input_ids = tokenizer.encode_qa(q, c, MAX_LEN_QA)
            self.starts, self.ends = find_answer_positions(q, c, a, tokenizer, MAX_LEN_QA)
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "start": self.starts[idx], "end": self.ends[idx]}

    def normalize_answer(s):
        s = s.lower()
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = ''.join(ch for ch in s if ch not in set(string.punctuation))
        s = ' '.join(s.split())
        return s

    def f1_score(pred, truth):
        pred_tok = normalize_answer(pred).split()
        truth_tok = normalize_answer(truth).split()
        common = Counter(pred_tok) & Counter(truth_tok)
        num_same = sum(common.values())
        if num_same == 0: return 0
        if len(pred_tok) == 0 or len(truth_tok) == 0: return 0
        precision = 1.0 * num_same / len(pred_tok)
        recall = 1.0 * num_same / len(truth_tok)
        return (2 * precision * recall) / (precision + recall)
        
    def evaluate_qa(model, dataset_raw, dataloader, tokenizer):
        model.eval()
        predictions = {}
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating QA")):
                input_ids = batch["input_ids"].to(DEVICE)
                start_logits, end_logits = model(input_ids)
                start_preds, end_preds = torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1)
                for i in range(len(input_ids)):
                    s, e = start_preds[i].item(), end_preds[i].item()
                    if e < s: e = s
                    batch_idx = idx * dataloader.batch_size + i
                    if batch_idx >= len(dataset_raw): continue
                    ex_id = dataset_raw[batch_idx]['id']
                    pred_tok_ids = input_ids[i][s:e+1].cpu().tolist()
                    predictions[ex_id] = tokenizer.decode(pred_tok_ids)
        f1 = exact_match = 0
        for ex in dataset_raw:
            ex_id = ex['id']
            truths = ex['answers']['text']
            pred = predictions.get(ex_id, "")
            f1 += max(f1_score(pred, gt) for gt in truths)
            exact_match += max(normalize_answer(pred) == normalize_answer(gt) for gt in truths)
        return f1 / len(dataset_raw), exact_match / len(dataset_raw)

    dataset = load_dataset("squad")
    train_ds_raw = dataset["train"].shuffle(seed=42).select(range(5000))
    val_ds_raw = dataset["validation"].select(range(1000))
    
    train_ds = SQuADDataset(train_ds_raw["question"], train_ds_raw["context"], train_ds_raw["answers"], tokenizer)
    val_ds = SQuADDataset(val_ds_raw["question"], val_ds_raw["context"], val_ds_raw["answers"], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    bert = create_bert_model_pytorch(tokenizer.get_vocab_size(), MAX_LEN_QA, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS)
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        pretrained_dict = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=DEVICE)
        model_dict = bert.state_dict()
        pretrained_pos_emb = pretrained_dict.get('embedding.pos_emb.weight')
        if pretrained_pos_emb is not None:
            model_pos_emb = model_dict['embedding.pos_emb.weight']
            len_copy = min(pretrained_pos_emb.shape[0], model_pos_emb.shape[0])
            model_pos_emb[:len_copy, :] = pretrained_pos_emb[:len_copy, :]
            model_dict['embedding.pos_emb.weight'] = model_pos_emb
            del pretrained_dict['embedding.pos_emb.weight']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        bert.load_state_dict(model_dict)
        print("Loaded pre-trained weights for QA model with resizing.")
    model = BERTForQuestionAnswering(bert).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1): # Reduced epochs for speed
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} SQuAD"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            starts, ends = batch["start"].to(DEVICE), batch["end"].to(DEVICE)
            start_logits, end_logits = model(input_ids)
            loss_fct = nn.CrossEntropyLoss()
            loss = (loss_fct(start_logits, starts) + loss_fct(end_logits, ends)) / 2
            loss.backward()
            optimizer.step()
    
    f1, em = evaluate_qa(model, val_ds_raw, val_loader, tokenizer)
    print(f"SQuAD F1: {f1:.4f}, Exact Match: {em:.4f}")
    return f1

def main():
    """Main function to run all fine-tuning tasks and print results."""
    tokenizer = get_tokenizer()
    results = {}
    results['Sentiment (Acc)'] = finetune_sentiment_classification(tokenizer)
    results['SNLI (Acc)'] = finetune_semantic_similarity(tokenizer)
    results['SQuAD (F1)'] = finetune_question_answering(tokenizer)

    print("\n\n" + "="*60)
    print("--- Custom PyTorch BERT Fine-Tuning Final Results ---")
    print("="*60)
    for task, score in results.items():
        print(f"{task:<25}: {score:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()