import torch
import yaml
import os
from model import Transformer, create_padding_mask, create_causal_mask
from tokenizers import Tokenizer as LibTokenizer

# 1. SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_model(checkpoint_path):
    model = Transformer(**config['model']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def translate(text, model, src_tok, tgt_tok, max_len=80):
    model.eval()
    
    # Encode Source
    src_ids = [2] + src_tok.encode(text).ids + [3]
    src_tensor = torch.tensor([src_ids]).to(device)
    src_mask = create_padding_mask(src_tensor, 0).to(device)
    
    # Pre-calculate Encoder Output
    src_emb = model.pos_encoding(model.src_embedding(src_tensor) * (model.d_model ** 0.5))
    enc_out = src_emb
    for layer in model.encoder:
        enc_out = layer(enc_out, src_mask)
        
    # Start with <bos>
    tgt_ids = [2]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_ids]).to(device)
        tgt_mask = create_causal_mask(tgt_tensor.size(1)).to(device)
        
        # Decoder logic
        tgt_emb = model.pos_encoding(model.tgt_embedding(tgt_tensor) * (model.d_model ** 0.5))
        dec_out = tgt_emb
        for layer in model.decoder:
            dec_out = layer(dec_out, enc_out, tgt_mask, src_mask)
            
        output = model.output_layer(model.final_norm(dec_out))
        
        # The last predicted word
        next_token = output[0, -1, :].argmax().item()
        tgt_ids.append(next_token)
        
        # If model predicts <eos>, stop
        if next_token == 3:
            break
            
    return tgt_tok.decode(tgt_ids)


if __name__ == "__main__":
    print("Loading model...")
    model = load_model("checkpoints/transformer_base_1M_32k.pt")
    
    src_tok = LibTokenizer.from_file("checkpoints/src_tok.json")
    tgt_tok = LibTokenizer.from_file("checkpoints/tgt_tok.json")
    
    print("\nTranslator Ready! Type 'q' to quit.")
    while True:
        german_text = input("\nGerman: ")
        if german_text.lower() == 'q':
            break
            
        english_text = translate(german_text, model, src_tok, tgt_tok)
        print(f"English: {english_text}")