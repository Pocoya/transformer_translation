import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from data import get_dataloaders
from model import Transformer, create_padding_mask, create_causal_mask

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def compute_perplexity(loss):
    """Perplexity = exp(loss) - measures prediction quality"""
    return math.exp(min(loss, 100)) # Cap at 100 to prevent overflow

def save_checkpoint(state, filename):
    """Saves everything needed to resume training exactly where we left off."""
    print(f"Saving full state to {filename}...")
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer, scheduler, scaler):
    """Loads all states to resume training."""
    print(f"Resuming from checkpoint: {filename}")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    scaler.load_state_dict(checkpoint['scaler_state'])
    return checkpoint['epoch'], checkpoint['history'], checkpoint['best_val_loss']


def train_epoch(model, loader, optimizer, criterion, device, epoch, config, history, scaler, scheduler):
    model.train()
    optimizer.zero_grad() # Reset gradients
    
    accum_steps = config['training'].get('accumulation_steps', 1)
    total_loss, total_acc = 0, 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # CREATE MASKS
        src_mask = create_padding_mask(batch['src'], 0).to(device)
        tgt_pad_mask = create_padding_mask(batch['tgt_input'], 0).to(device)
        tgt_causal_mask = create_causal_mask(batch['tgt_input'].size(1)).to(device)
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        
        # FORWARD PASS with Mixed Precision
        with torch.amp.autocast('cuda', enabled=config['training'].get('use_amp', True)):
            output = model(batch['src'], batch['tgt_input'], src_mask, tgt_mask)
            raw_loss = criterion(
                output.reshape(-1, output.size(-1)), 
                batch['tgt_output'].reshape(-1)
            )
            # Scale the loss by accum_steps to ensure the gradient average is correct
            loss = raw_loss / accum_steps
        
        # BACKWARD PASS with Scaler
        scaler.scale(loss).backward()
        
        # UPDATE WEIGHTS
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() # Clear gradients for the next accumulation cycle
            scheduler.step()     
        
        # ACCURACY (Ignore padding tokens)
        preds = output.argmax(-1).reshape(-1)
        targets = batch['tgt_output'].reshape(-1)
        non_pad_mask = (targets != 0)
        correct = (preds == targets) & non_pad_mask
        acc = correct.sum().item() / (non_pad_mask.sum().item() + 1e-9)
        
        total_loss += raw_loss.item()
        total_acc += acc
        num_batches += 1
        
        if batch_idx % config['logging']['log_every'] == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, "
                  f"Loss: {raw_loss.item():.4f}, Acc: {acc:.3f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
    return total_loss / num_batches, total_acc / num_batches

@torch.no_grad()
def validate_epoch(model, loader, criterion, device, config):
    model.eval()
    total_loss, total_acc = 0, 0
    num_batches = 0
    
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Validation, same masking logic as training
        src_mask = create_padding_mask(batch['src'], 0).to(device)
        tgt_pad_mask = create_padding_mask(batch['tgt_input'], 0).to(device)
        tgt_causal_mask = create_causal_mask(batch['tgt_input'].size(1)).to(device)
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        
        output = model(batch['src'], batch['tgt_input'], src_mask, tgt_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), 
                        batch['tgt_output'].reshape(-1))
        
        # Validation Accuracy, ignore padding
        preds = output.argmax(-1).reshape(-1)
        targets = batch['tgt_output'].reshape(-1)
        non_pad_mask = (targets != 0)
        correct = (preds == targets) & non_pad_mask
        acc = correct.sum().item() / non_pad_mask.sum().item()
        
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches

def plot_training_curves(history, config, plot_dir="plots"):
    """Plot at the end of epochs"""
    epoch = len(history['train_loss'])
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Cross Entropy Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([math.exp(l) for l in history['train_loss']], label='Train')
    plt.plot([math.exp(l) for l in history['val_loss']], label='Val')
    plt.title('Perplexity')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Token Accuracy (No Padding)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"results_epoch_{epoch}.png"))
    plt.savefig(os.path.join(plot_dir, "latest.png"))
    plt.close()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_dir = "checkpoints"
    plot_dir = "plots"
    for d in [model_dir, plot_dir]:
        os.makedirs(d, exist_ok=True)

    # DATA LOADERS
    train_loader, val_loader, _, src_tokenizer, tgt_tokenizer = get_dataloaders(
        batch_size=config['training']['batch_size'])

    # Save tokenizers for translation
    src_tokenizer.tokenizer.save("checkpoints/src_tok.json")
    tgt_tokenizer.tokenizer.save("checkpoints/tgt_tok.json")
    
    # MODEL, LOSS, OPTIMIZER
    model = Transformer(**config['model']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=0, 
        label_smoothing=config['training'].get('label_smoothing', 0.1)
    )
    
    optimizer = AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=config['training'].get('use_amp', True))
    
    # SCHEDULER
    accum_steps = config['training'].get('accumulation_steps', 1)
    effective_steps_per_epoch = len(train_loader) // accum_steps
    total_optimization_steps = effective_steps_per_epoch * config['training']['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=total_optimization_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # HISTORY & TRACKING
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    start_epoch = 0
    
    # EARLY STOPPING SETUP
    early_stop_counter = 0
    patience = config['training'].get('early_stopping_patience', 4)

    # RESUME TRAINING?
    resume_path = config['training'].get('resume_from')
    if resume_path and os.path.exists(resume_path):
        last_epoch, history, best_val_loss = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
        start_epoch = last_epoch + 1
    
    # TRAINING LOOP
    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\n{'='*30}")
        print(f"EPOCH {epoch+1}/{config['training']['epochs']}")
        print(f"{'='*30}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch, config, history, scaler, scheduler
        )
        
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, config)
        
        # Record results
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"\nResults - Epoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc:  {train_acc:.3f} | Val Acc:  {val_acc:.3f}")
        
        # SAVE FULL STATE
        full_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss
        }
        save_checkpoint(full_state, os.path.join(model_dir, "last_state.tar"))

        # BEST MODEL & EARLY STOPPING LOGIC
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_path = os.path.join(model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (Loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epoch(s).")
            
        # PERIODIC CHECKPOINT
        if (epoch + 1) % config['training']['save_every'] == 0:
            ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
        # UPDATE PLOTS
        plot_training_curves(history, config, plot_dir=plot_dir)

        # TERMINATION CHECK
        if early_stop_counter >= patience and epoch > 5:
            print(f"\nEARLY STOPPING: No improvement for {patience} epochs. Saturation reached.")
            break

    print(f"\nTraining Finished! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()