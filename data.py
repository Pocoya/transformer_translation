import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

def clean_logic(text):
    """
    Cleans data: removes too short/long sentences, 
    removes identical translations, and checks length ratios.
    """
    src = text['translation']['de'].strip()
    tgt = text['translation']['en'].strip()

    # Length check
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    if not (3 <= src_len <= 80 and 3 <= tgt_len <= 80):
        return False
    
    # Ratio check (prevents dirty data where one side is much longer)
    if max(src_len, tgt_len) / min(src_len, tgt_len) > 2.5:
        return False
        
    # Identity check (prevents model learning to just copy input)
    if src.lower() == tgt.lower():
        return False
        
    return True

class FastBPETokenizer:
    """Wrapper for the Rust-based BPE Tokenizer"""
    def __init__(self, vocab_size=16000):
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.vocab_size = vocab_size

    def train(self, texts):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        self.tokenizer.train_from_iterator(texts, trainer)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        # Filter out special tokens for clean output
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode([i for i in ids if i not in [PAD_IDX, BOS_IDX, EOS_IDX]])

class PreTokenizedDataset(Dataset):
    def __init__(self, raw_dataset, src_tokenizer, tgt_tokenizer, max_len=80):
        self.src_tensors = []
        self.tgt_tensors = []

        print(f"Tokenizing {len(raw_dataset)} samples...")
        for item in raw_dataset:
            # Encode
            s_ids = src_tokenizer.encode(item['translation']['de'])
            t_ids = tgt_tokenizer.encode(item['translation']['en'])

            # Add BOS/EOS and truncate
            s_ids = [BOS_IDX] + s_ids[:max_len-2] + [EOS_IDX]
            t_ids = [BOS_IDX] + t_ids[:max_len-2] + [EOS_IDX]

            # Pad
            s_pad = s_ids + [PAD_IDX] * (max_len - len(s_ids))
            t_pad = t_ids + [PAD_IDX] * (max_len - len(t_ids))

            self.src_tensors.append(torch.tensor(s_pad, dtype=torch.long))
            self.tgt_tensors.append(torch.tensor(t_pad, dtype=torch.long))

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, idx):
        src = self.src_tensors[idx]
        tgt = self.tgt_tensors[idx]
        
        # tgt_input: <bos> I love cats <pad>
        # tgt_output: I love cats <eos> <pad>
        return {
            'src': src,
            'tgt_input': tgt[:-1],
            'tgt_output': tgt[1:]
        }

def get_dataloaders(batch_size=256, max_len=80, vocab_size=32000):
    print("Loading WMT14 dataset...")
    raw_dataset = load_dataset("wmt/wmt14", "de-en")

    print("Filtering 1M samples...")
    train_raw = raw_dataset["train"].select(range(1000000)).filter(clean_logic)
    val_raw = raw_dataset['validation'].filter(clean_logic)
    test_raw = raw_dataset['test'].select(range(2000)).filter(clean_logic)

    print(f"Dataset sizes: Train: {len(train_raw)} | Val: {len(val_raw)}")

    print("Training Tokenizers (BPE)...")
    src_tokenizer = FastBPETokenizer(vocab_size)
    tgt_tokenizer = FastBPETokenizer(vocab_size)

    def get_texts(ds, lang):
        for i in range(len(ds)):
            yield ds[i]['translation'][lang]

    src_tokenizer.train(get_texts(train_raw, 'de'))
    tgt_tokenizer.train(get_texts(train_raw, 'en'))

    print("Creating Tensors (Pre-tokenizing)...")
    train_ds = PreTokenizedDataset(train_raw, src_tokenizer, tgt_tokenizer, max_len)
    val_ds = PreTokenizedDataset(val_raw, src_tokenizer, tgt_tokenizer, max_len)
    test_ds = PreTokenizedDataset(test_raw, src_tokenizer, tgt_tokenizer, max_len)

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer
