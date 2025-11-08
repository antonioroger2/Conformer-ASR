import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import jiwer
import os
import argparse

# Import modules from the new structure
from configs.config import cfg
from data_prep.dataset import LibriSpeechDataset, collate_fn, tokenizer
from models.conformer import ConformerCTC
from utils.checkpoints import count_parameters, save_checkpoint, load_checkpoint

def evaluate_model(model, test_loader, device):
    """Evaluates the model and computes the Word Error Rate (WER)."""
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None:
                continue

            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)

            # Get log probabilities from the model
            log_probs = model(features, feature_lengths) # (B, T', V)
            
            # Simple greedy decoding
            predictions = torch.argmax(log_probs, dim=-1) # (B, T')

            for i, pred_seq in enumerate(predictions):
                length = feature_lengths[i].item()
                pred_seq = pred_seq[:length].cpu().numpy()

                # CTC greedy path-merging and blank removal
                decoded = []
                prev_token = None
                for token in pred_seq:
                    if token != 0 and token != prev_token: # 0 is the <blank> token
                        decoded.append(token)
                    prev_token = token

                predicted_text = tokenizer.decode(decoded)
                all_predictions.append(predicted_text)

            # Extract reference transcripts
            transcript_lengths = batch['transcript_lengths']
            transcripts = batch['transcripts']
            start_idx = 0
            for length in transcript_lengths:
                ref_tokens = transcripts[start_idx:start_idx + length].cpu().numpy()
                # Use the raw index to char mapping for the reference
                reference_text = tokenizer.decode(ref_tokens) 
                all_references.append(reference_text)
                start_idx += length

    # Calculate Word Error Rate
    wer_score = jiwer.wer(all_references, all_predictions)

    print("\n" + "="*20 + " Evaluation Results " + "="*20)
    print(f"WER: {wer_score:.4f}")
    print(f"Evaluated on {len(all_predictions)} samples")

    print("\nSample predictions:")
    for i in range(min(5, len(all_predictions))):
        print(f"Reference: {all_references[i]}")
        print(f"Predicted: {all_predictions[i]}")

    return wer_score

def train_model(config):
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    
    # 1. Data Loading
    print("Loading datasets...")
    train_dataset = LibriSpeechDataset(config.train_data_path, config.wav_len, is_train=True)
    test_dataset = LibriSpeechDataset(config.test_data_path, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size // 2, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # 2. Model Setup
    model = ConformerCTC(config)
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)

    print(f"Model params: {count_parameters(model)/1e6:.2f}M")

    # 3. Loss, Optimizer, Scheduler
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr, weight_decay=1e-6)

    total_steps = len(train_loader) * config.nepochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.base_lr, total_steps=total_steps,
        pct_start=config.pct_start, div_factor=config.div_factor,
        final_div_factor=config.final_div_factor, anneal_strategy=config.anneal_strategy
    )

    # Mixed Precision Setup
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 4. Load Checkpoint
    start_epoch, best_loss = 0, float('inf')
    checkpoint_path = os.path.join(config.save_dir, 'latest_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        print(f"Resuming from epoch {start_epoch}, best loss {best_loss:.4f}")

    # 5. Training Loop
    print("\n" + "=" * 50)
    print("Starting Training Loop")
    print("=" * 50 + "\n")
    for epoch in range(start_epoch, config.nepochs):
        model.train()
        collate_fn.training = True # Enable SpecAugment in collate_fn
        total_loss, num_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.nepochs}")

        for batch in pbar:
            if batch is None:
                continue
            optimizer.zero_grad()
            
            # Prepare data
            features = batch['features'].to(device)
            feature_lengths = batch['feature_lengths'].to(device)
            transcripts = batch['transcripts'].to(device)
            transcript_lengths = batch['transcript_lengths'].to(device)

            # Forward pass with autocast
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                log_probs = model(features, feature_lengths)
                log_probs = log_probs.transpose(0,1) # CTC loss expects (T, B, V)
                loss = ctc_loss(log_probs, transcripts, feature_lengths, transcript_lengths)

            if torch.isnan(loss):
                continue

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update logging
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"\nEpoch {epoch+1} avg_loss: {avg_loss:.4f}")

        # Save Checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': min(best_loss, avg_loss)
        }
        save_checkpoint(checkpoint, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(checkpoint, os.path.join(config.save_dir, 'best_model.pth'))
            print(f"New best model saved with loss {best_loss:.4f}")

    # 6. Final Model Save and Evaluation
    final_model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    # Save to the generic path for test.py to use
    torch.save(final_model_state, config.generic_model_path) 
    print(f"\nFinal model saved at {config.generic_model_path}")

    # Evaluate on the test set
    collate_fn.training = False # Disable SpecAugment
    wer_score = evaluate_model(model, test_loader, device)

    return model, wer_score

def main():
    parser = argparse.ArgumentParser(description="Conformer-CTC ASR Training")
    parser.add_argument('--train_path', type=str, default=cfg.train_data_path,
                        help="Path to the training data directory (e.g., train-clean-100).")
    parser.add_argument('--test_path', type=str, default=cfg.test_data_path,
                        help="Path to the test data directory (e.g., test-clean).")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    cfg.train_data_path = args.train_path
    cfg.test_data_path = args.test_path
    
    if not os.path.exists(cfg.train_data_path):
        print(f"Error: Training data path not found: {cfg.train_data_path}")
        return
    if not os.path.exists(cfg.test_data_path):
        print(f"Error: Test data path not found: {cfg.test_data_path}")
        return
    
    print("=" * 50)
    print("Training Conformer-CTC ASR Model")
    print("=" * 50)
    
    _, final_wer = train_model(cfg)

    print(f"\nTraining completed!")
    print(f"Final WER: {final_wer:.4f}")

if __name__ == "__main__":
    main()