import torch
import os
import random
import torchaudio
from jiwer import wer, cer
import argparse

# Import modules from the new structure
from configs.config import cfg
from data_prep.dataset import mel_spectrogram, normalize_spec
from models.conformer import ConformerCTC
from data_prep.dataset import SimpleTokenizer, clean_text 

# Re-initialize tokenizer to be available in test.py
tokenizer = SimpleTokenizer()
cfg.vocab_size = tokenizer.get_vocab_size()

def get_ground_truth(audio_path, data_dir):
    """
    Finds the corresponding transcript for an audio file from the .trans.txt files.
    Note: The original notebook's logic for finding the transcript file was slightly
    flawed for a general path structure, so this is a simplified version based on 
    LibriSpeech directory structure.
    """
    # Navigate up to the directory containing the .trans.txt file
    # LibriSpeech structure: .../speaker_id/chapter_id/audio_file.flac
    # Transcript file is: .../speaker_id/chapter_id.trans.txt
    
    base_dir = os.path.dirname(audio_path) # e.g., .../test-clean/speaker_id/chapter_id
    parent_dir = os.path.dirname(base_dir) # e.g., .../test-clean/speaker_id
    chapter_id = os.path.basename(base_dir)
    
    # Construct the path to the transcript file
    # Note: In LibriSpeech, the transcript file is often named after the Chapter ID 
    # and is located *inside* the Chapter ID folder for convenience in the ASR repo, 
    # but the original code expected it up one level for a specific file naming.
    # We will use the original notebook's simplified approach which expected 
    # transcript_file = os.path.join(base_dir, f"{speaker_id}.trans.txt")
    
    # A safer approach is to find the file that contains the utt_id in the transcript file.
    # The original notebook's logic was:
    # speaker_id = os.path.basename(parent_dir) # Incorrectly fetches the *parent* of the chapter_id folder
    # transcript_file = os.path.join(base_dir, f"{speaker_id}.trans.txt") # This path construction is usually wrong
    
    # Let's search for any .txt file in the chapter directory. For LibriSpeech, it's typically 'chapter_id.trans.txt' or similar.
    # The original notebook's logic was wrong, but we'll try to replicate the intent:
    
    # The audio ID is the full name (e.g., 100-121-0000)
    utt_id = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Look for any .txt file in the same directory as the audio file
    for f in os.listdir(base_dir):
        if f.endswith('.txt'):
            transcript_file = os.path.join(base_dir, f)
            with open(transcript_file, "r") as tr_f:
                for line in tr_f:
                    if line.startswith(utt_id):
                        # Returns the cleaned (uppercase) transcript for comparison
                        # The WER/CER call in the original notebook used .lower(), which is inconsistent
                        # with the training (A-Z, ') but common in ASR evaluation. 
                        # I will return the original-notebook-style lower-case text.
                        return " ".join(line.strip().split(" ")[1:]).lower()
    return None

@torch.no_grad()
def transcribe_audio(model, audio_path, cfg, tokenizer, device):
    """Performs transcription on a single audio file."""
    model.eval()
    
    # 1. Load and preprocess audio
    wav, sr = torchaudio.load(audio_path)
    if sr != cfg.sample_rate:
        wav = torchaudio.transforms.Resample(sr, cfg.sample_rate)(wav)

    features = mel_spectrogram(wav, cfg.sample_rate, n_mels=cfg.n_mels)
    features_norm, _ = normalize_spec(features.squeeze(0))
    features_norm = features_norm.unsqueeze(0).to(device)

    # 2. Forward pass
    log_probs = model(features_norm) # (1, T', V)

    # 3. Greedy decoding (same logic as in evaluate_model)
    predictions = torch.argmax(log_probs, dim=-1).squeeze(0) # (T')

    decoded = []
    prev_token = None
    for token in predictions:
        token = token.item()
        if token != 0 and token != prev_token: # 0 is the <blank> token
            decoded.append(token)
        prev_token = token

    # 4. Decode to text
    return tokenizer.decode(decoded)

def main():
    parser = argparse.ArgumentParser(description="Conformer-CTC ASR Testing")
    parser.add_argument('--model_path', type=str, default=cfg.generic_model_path,
                        help="Path to the saved final model .pth file.")
    parser.add_argument('--data_dir', type=str, default=cfg.test_data_path,
                        help="Path to the test data directory (e.g., test-clean) for sampling audio.")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="Number of random samples to transcribe.")
    
    args = parser.parse_args()
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    try:
        model = ConformerCTC(cfg)
        state_dict = torch.load(args.model_path, map_location=dev)
        # Handle DataParallel state dict if necessary (strip 'module.')
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        model = model.to(dev)
        model.eval()
        print(f"Model successfully loaded from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been trained and the path is correct.")
        return

    # 2. Find and Sample Audio Files
    audio_files = []
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if f.endswith(".flac"):
                audio_files.append(os.path.join(root, f))
                
    if not audio_files:
        print(f"Error: No .flac files found in {args.data_dir}")
        return

    num_samples = min(args.num_samples, len(audio_files))
    random_samples = random.sample(audio_files, num_samples)
    print(f"Sampling and transcribing {num_samples} audio files...")

    # 3. Transcription and Evaluation
    total_wer, total_cer = 0, 0
    valid_samples = 0

    print("\n" + "="*20 + " Transcription Results " + "="*20)
    for idx, audio_path in enumerate(random_samples, 1):
        truth = get_ground_truth(audio_path, args.data_dir)
        if not truth:
            # print(f"Warning: Could not find ground truth for {os.path.basename(audio_path)}")
            continue
        
        predicted = transcribe_audio(model, audio_path, cfg, tokenizer, dev)
        
        # WER/CER calculation expects the *reference* to be lower-case (as used in the original notebook)
        # and the *predicted* text to be the output of the model (uppercase, with special chars).
        # We need to clean the predicted output to match the reference style for comparison.
        # The tokenizer.decode outputs uppercase (A-Z, space, ') which is compared against the lower-case truth.
        # Let's convert the prediction to lowercase to match the original notebook's evaluation style.
        predicted_lower = predicted.lower() 

        sample_wer = wer(truth, predicted_lower)
        sample_cer = cer(truth, predicted_lower)
        
        total_wer += sample_wer
        total_cer += sample_cer
        valid_samples += 1

        print(f"\n[{idx}] FILE: {os.path.basename(audio_path)}")
        print(f"TRUTH:     {truth}")
        print(f"PREDICTED: {predicted}") # Display the raw output (UPPERCASE)
        print(f"WER: {sample_wer:.3f} | CER: {sample_cer:.3f}")
    
    if valid_samples > 0:
        print("\n" + "="*50)
        print(f"Mean WER (on {valid_samples} samples): {total_wer / valid_samples:.3f}")
        print(f"Mean CER (on {valid_samples} samples): {total_cer / valid_samples:.3f}")
    else:
        print("\nCould not find ground truth for any sampled audio files. Cannot report mean WER/CER.")

if __name__ == "__main__":
    main()
