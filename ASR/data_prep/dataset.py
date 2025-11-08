import torch
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import numpy as np
import os
import random
import re
from configs.config import cfg # Relative import

def clean_text(text):
    """Cleans the transcript text for tokenization."""
    text = text.upper()
    # Remove characters not in the vocab (A-Z, 0-9, space, hyphen, apostrophe)
    text = re.sub('[^A-Z\\s\\-\']+', '', text)
    return text.strip()

def mel_spectrogram(audio, sample_rate=16000, hop_length=160, win_length=400, n_mels=80):
    """Computes log Mel spectrogram from raw audio."""
    audio_to_mel = AT.Spectrogram(
        hop_length=hop_length, win_length=win_length, n_fft=win_length, power=1.0, normalized=False,
    ).to(audio.device)

    mel_scale = AT.MelScale(
        sample_rate=sample_rate, n_stft=win_length // 2 + 1, n_mels=n_mels, f_min=0.,
        f_max=sample_rate//2, norm="slaney", mel_scale="slaney",
    ).to(audio.device)

    spec = audio_to_mel(audio)
    mel = mel_scale(spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    # Output shape: (Batch, Time, Mel_dim)
    return mel.permute(0, 2, 1) 

def normalize_spec(x):
    """Normalizes the spectrogram features."""
    mean = x.mean()
    std = x.std() + 1e-5
    return (x - mean) / std, mean

def spec_augment(x, freq_mask=27, time_mask_ratio=0.05):
    """Applies simple SpecAugment for regularization during training."""
    # Check if a 'training' attribute has been set on the function
    if not hasattr(spec_augment, 'training') or not spec_augment.training:
        return x

    x = x.clone()
    t_size, f_size = x.shape

    # Frequency masking
    if random.random() < 0.5:
        f = random.randint(1, min(freq_mask, f_size//4))
        f0 = random.randint(0, f_size - f)
        x[:, f0:f0+f] = x.mean()

    # Time masking
    if random.random() < 0.5:
        t = random.randint(1, max(1, int(t_size * time_mask_ratio)))
        t0 = random.randint(0, max(1, t_size - t))
        x[t0:t0+t, :] = x.mean()

    return x

class SimpleTokenizer:
    """A simple character-level tokenizer."""
    def __init__(self):
        # Vocab: <blank> (0), space (1), A-Z (2-27), apostrophe (28)
        self.vocab = ['<blank>', ' '] + [chr(i) for i in range(ord('A'), ord('Z')+1)] + ["'"]
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text):
        """Converts a string to a list of token indices."""
        return [self.char_to_idx.get(char, 0) for char in text]

    def decode(self, indices):
        """Converts a list of token indices back to a string."""
        # Note: CTC decoding post-processing (removing blanks and repeats) happens elsewhere.
        # This is just a simple index-to-character conversion.
        return ''.join([self.idx_to_char.get(idx, '') for idx in indices if idx != 0])

    def get_vocab_size(self):
        return len(self.vocab)

tokenizer = SimpleTokenizer()
# Update the global config with the actual vocabulary size
cfg.vocab_size = tokenizer.get_vocab_size()


class LibriSpeechDataset(Dataset):
    """Dataset class for LibriSpeech ASR."""
    def __init__(self, data_path, max_length=None, is_train=True):
        self.data_path = data_path
        self.max_length = max_length
        self.is_train = is_train
        self.data = []

        # Find all .flac and corresponding .txt files
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.txt') and not file.startswith('.'):
                    txt_path = os.path.join(root, file)
                    with open(txt_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                audio_id, transcript = parts
                                # Assuming audio_id maps directly to the .flac file in the same directory
                                audio_path = os.path.join(root, audio_id + '.flac') 
                                if os.path.exists(audio_path):
                                    self.data.append({
                                        'audio_path': audio_path,
                                        'transcript': clean_text(transcript)
                                    })

        print(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            # Load audio
            wav, sr = torchaudio.load(item['audio_path'])
            if sr != cfg.sample_rate:
                wav = AT.Resample(sr, cfg.sample_rate)(wav)

            # Max length cropping for training
            if self.is_train and self.max_length:
                max_samples = int(cfg.sample_rate * cfg.wav_len)
                if wav.shape[1] > max_samples:
                    start = random.randint(0, wav.shape[1] - max_samples)
                    wav = wav[:, start:start + max_samples]

            # Compute features (Mel Spectrogram)
            features = mel_spectrogram(wav, cfg.sample_rate, n_mels=cfg.n_mels)
            features = features.squeeze(0) # (Time, Mel_dim)

            return features, item['transcript'], item['audio_path']

        except Exception as e:
            # print(f"Error loading {item['audio_path']}: {e}")
            return None, None, None

def collate_fn(batch):
    """
    Collate function for the DataLoader.
    Pads features, normalizes, applies augmentation, and encodes transcripts.
    """
    # Filter out samples that failed to load
    batch = [(f, t, p) for f, t, p in batch if f is not None]
    if len(batch) == 0:
        return None

    features, transcripts, paths = zip(*batch)

    # 1. Feature Preprocessing (Normalization and Augmentation)
    feature_batch = []
    for feat in features:
        feat_norm, _ = normalize_spec(feat)
        # Apply SpecAugment if in training mode (set by train.py)
        if hasattr(collate_fn, 'training') and collate_fn.training:
            spec_augment.training = True
            feat_norm = spec_augment(feat_norm)
        else:
            spec_augment.training = False
        feature_batch.append(feat_norm)

    # Pad features to the longest in the batch
    packed_features = pack_sequence(feature_batch, enforce_sorted=False)
    padded_features, feature_lengths = pad_packed_sequence(packed_features, batch_first=True)

    # 2. Transcript Encoding
    encoded_transcripts = []
    transcript_lengths = []
    for transcript in transcripts:
        encoded = tokenizer.encode(transcript)
        encoded_transcripts.extend(encoded)
        transcript_lengths.append(len(encoded))

    # 3. Downsample feature lengths (output length for CTC loss)
    # The feature length is reduced by cfg.downsample (e.g., 4) in the DownSampler module
    downsampled_feature_lengths = torch.ceil(feature_lengths.float() / cfg.downsample).int()
    
    return {
        'features': padded_features, # (B, T, F)
        'feature_lengths': downsampled_feature_lengths, # (B)
        'transcripts': torch.tensor(encoded_transcripts), # (Sum of all sequence lengths)
        'transcript_lengths': torch.tensor(transcript_lengths), # (B)
        'paths': paths
    }