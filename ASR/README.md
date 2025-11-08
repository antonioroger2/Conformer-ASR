# 🎙️ Conformer-ASR: Architectural and Mathematical Intuition

The system follows an end-to-end approach, integrating a powerful **Conformer Encoder** with a **Connectionist Temporal Classification (CTC) Decoder** for alignment-free sequence prediction.


## Overall Architecture

The ASR pipeline consists of four main stages:

1. **Feature Extraction** – Raw audio → Mel Spectrogram features  
2. **Downsampler Convolutional Frontend** – Reduces sequence length & projects features  
3. **Conformer Encoder** – Captures local + global dependencies  
4. **Alignment & Decoding** – Frame-wise predictions → Text transcription (via CTC)


## Input Processing and Feature Extraction

### A. Mel Spectrogram Transformation

- Input audio: **16 kHz waveform**
- Parameters:  
  - FFT window (`N_fft`) = **400**  
  - Hop length = **160** (→ 10 ms per frame since 160 / 16000 = 0.01 s)  
  - Mel bins (`F`) = **80**
- Features are normalized (mean/std) before passing to the model.

### B. Downsampler Convolutional Frontend

This module serves two purposes:

1. **Sequence Reduction** → $T' = T / 4$  
2. **Feature Projection** → $F = 80 \rightarrow D_{model} = 512$

Achieved with **two stacked 2D convolutional layers (stride=2)** followed by a **linear projection**.

| Input Shape | Output Shape |
|--------------|---------------|
| (B, T, F=80) | (B, T′=T/4, D=512) |



## 2️⃣ The Conformer Encoder

The encoder is a stack of **12 Conformer Blocks**.  
Each block integrates **Feed-Forward**, **Multi-Head Self-Attention**, and **Convolution** modules.

### 📐 Conformer Block Equations

Let $x$ be the input tensor:

$$
\begin{aligned}
x' &= x + \frac{1}{2} \cdot \text{FFN}(x) && \text{(First Feed-Forward)} \\
x'' &= x' + \text{MHA}(x') && \text{(Multi-Head Self-Attention)} \\
x''' &= x'' + \text{Conv}(x'') && \text{(Convolution Module)} \\
x_{\text{out}} &= x''' + \frac{1}{2} \cdot \text{FFN}(x''') && \text{(Second Feed-Forward)}
\end{aligned}
$$

Each layer uses **pre-norm residual connections**.



### A. Feed-Forward Modules (FF1 and FF2)

$$
\text{FFN}(x) = \text{Linear}_{D \to D} \left( \text{Dropout} \big( \text{ReLU} \big( \text{Linear}_{D \to 4D} (\text{LayerNorm}(x)) \big) \big) \right)
$$

- Expansion factor: $D_{ff} = 4D_{model} = 2048$
- Input/Output: (B, T′, 512)
- Output scaled by **0.5** before residual connection



### B. Multi-Head Self-Attention (MHA)

- Heads: **8**  
- $D_{model} = 512 \Rightarrow D_k = D_{model} / H = 64$

The attention operation:

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left( \frac{Q K^T}{\sqrt{64}} \right)V
$$

$Q, K, V$ are linear projections of $x$;  
outputs from all 8 heads are concatenated and projected back to $D_{model}$.


### C. Convolution Module

Steps inside the Conv Module:

1. **LayerNorm + Transpose** → $(B, T′, D) \to (B, D, T′)$  
2. **Pointwise Conv1 (1×1)**: $D \to 2D$ (512 → 1024)  
   - Followed by **GLU** activation → halves back to 512  
3. **Depthwise Conv (kernel=31)**: groups = $D_{model}$  
   - Captures local temporal patterns  
4. **BatchNorm + SiLU (Swish)** activation  
5. **Pointwise Conv2 (1×1)**: $512 \to 512$


## 3️⃣ Alignment and Decoding Head

### A. Classifier Head

The encoder output passes through **LayerNorm** and a final **Linear** layer:

$$
\text{Logits} = \text{Linear}_{D_{model} \to V}(\text{LayerNorm}(x_{encoder}))
$$

- Input: (B, T′, 512)  
- Output: (B, T′, V), where **V = 29** (A–Z, space, apostrophe, `<blank>`)

A **LogSoftmax** converts logits to frame-wise probabilities.

### B. CTC Decoding

Training uses **CTC Loss**, which aligns speech frames with text automatically.  
Inference uses **Greedy Decoding**:

1. Take $\text{argmax}$ at each frame  
2. Remove consecutive duplicates  
3. Remove `<blank>` tokens  

✅ Output → clean text transcription


## Model Summary

| Module | Key Components | Purpose |
|---------|----------------|----------|
| **Frontend** | 2× Conv2D + Linear | Downsample and project features |
| **Encoder** | 12× Conformer Blocks | Learn temporal + contextual features |
| **FF Module** | Linear → ReLU → Dropout → Linear | Long-range transformations |
| **MHA** | 8-head scaled dot-product attention | Global context modeling |
| **Conv Module** | GLU + Depthwise Conv + SiLU | Local feature extraction |
| **CTC Head** | LayerNorm + Linear + LogSoftmax | Frame-to-character mapping |



## 🧠 Symbols

| Symbol | Meaning |
|---------|----------|
| $B$ | Batch size |
| $T$ | Input sequence length |
| $T'$ | Downsampled length ($T / 4$) |
| $F$ | Feature dimension (Mel bins = 80) |
| $D_{model}$ | Model hidden size (512) |
| $V$ | Vocabulary size (29) |

📚 **References**
- [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)
- [Connectionist Temporal Classification (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
