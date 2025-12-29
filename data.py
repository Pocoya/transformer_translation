from collections import Counter
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch

dataset = load_dataset("wmt/wmt14", "de-en")

train_dataset = dataset["train"].select(range(100_000))
val_dataset = dataset['validation']
test_dataset = dataset['test'].select(range(2000))

def clean(text):
    src = text['translation']['de'].strip()
    tgt = text['translation']['en'].strip()

    if 3 <= len(src.split()) <= 80 and 3 <= len(tgt.split()) <= 80:
        return {'translation': {'de': src, 'en': tgt}}
    return None


class Tokenizer:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
    
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + [word for word, _ in counter.most_common(self.vocab_size - 4)]
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>'] + [word for word, _ in counter.most_common(self.vocab_size - 4)]

        self.word2idx = {word: i for i, word in enumerate(vocab)}
        self.idx2word = {i: word for word, i in self.word2idx.items()}
    

    def encode(self, text, max_len=80):
        tokens = ['<bos>'] + text.lower().split() + ['<eos>']
        ids = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        if len(ids) < max_len:
            ids += [self.word2idx['<pad>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return torch.tensor(ids)

    def decode(self, ids):
        words = [self.idx2word[id.item()] for id in ids if id.item() not in [0, 2, 3]]
        return ' '.join(words)
    

class Dataset(Dataset):
    def __init__(self, dataset, src_tokenizer, tgt_tokenizer, max_len=80):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item['translation']['de']
        tgt_text = item['translation']['en']

        src_ids = self.src_tokenizer.encode(src_text, self.max_len)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, self.max_len)

        # Match target input and output, x[0] -> y[0] so x[0] != y[0]
        # x: input from 0 to last -1 to always have a token to predict
        # y: output from 1 to last to always predict the next token
        return {
            'src': src_ids,
            'tgt_input': tgt_ids[:-1],
            'tgt_output': tgt_ids[1:]
        }


def get_dataloaders(batch_size=64, max_len=80):
    print("Cleaning datasets...")
    train_data = train_dataset.filter(clean)
    val_data = val_dataset.filter(clean)
    test_data = test_dataset.filter(clean)

    print("Datasets cleaned.")
    print(f"Dataset sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print("Building tokenizers...")
    all_src = [item['translation']['de'] for item in train_data]
    all_tgt = [item['translation']['en'] for item in train_data]

    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()
    src_tokenizer.build_vocab(all_src)
    tgt_tokenizer.build_vocab(all_tgt)

    print(f"Vocab sizes: src={len(src_tokenizer.word2idx)}, tgt={len(tgt_tokenizer.word2idx)}")

    print("Creating datasets...")
    train_ds = Dataset(train_data, src_tokenizer, tgt_tokenizer, max_len)
    val_ds = Dataset(val_data, src_tokenizer, tgt_tokenizer, max_len)
    test_ds = Dataset(test_data, src_tokenizer, tgt_tokenizer, max_len)

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer = get_dataloaders()

    batch = next(iter(train_loader))
    print("\n✅ SUCCESS! Tokenization works:")
    print(f"src shape: {batch['src'].shape}")
    print(f"tgt_input shape: {batch['tgt_input'].shape}")
    print(f"tgt_output shape: {batch['tgt_output'].shape}")

    print("\nExample batch:")
    print("German:", src_tokenizer.decode(batch['src'][0]))
    print("English input:", tgt_tokenizer.decode(batch['tgt_input'][0]))

