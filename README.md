## Overview

This project implements **biometric detection using speech** with the **ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)** architecture.  
Speech-based biometrics provide a reliable and user-friendly way to verify and identify individuals based on their unique vocal characteristics.  

The **ECAPA-TDNN** model improves upon standard TDNN-based speaker verification by:
- Using **Res2Net modules** to capture multi-scale features.
- Applying **channel attention mechanisms** to emphasize important spectral features.
- Aggregating frame-level representations into a robust speaker embedding.

## Features

- 🎯 **State-of-the-art Model**: Utilizes ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN) from SpeechBrain for high-performance speaker embedding.
- 🔄 **Incremental Processing**: Designed to handle large audio datasets by appending extracted embeddings and labels to existing files, supporting resume capabilities.
- 📊 **Flexible Data Loading**: Processes `.tsv` and `.csv` metadata files to link audio clips to speaker information.
- 🎵 **Audio Preprocessing**: Includes audio resampling and fixed-duration segmentation to prepare clips for model input.
- 💾 **Efficient Storage**: Stores generated embeddings and labels as NumPy arrays (`.npy`) for efficient storage and retrieval.
- 🔍 **Speaker Recognition Ready**: Generated embeddings are directly suitable for downstream tasks such as speaker verification and identification.

## Project Structure

```
└── chiragbansal511-biometrics_detection_from_speech/
    ├── Code/
    │   ├── code1.ipynb
    │   └── code2.ipynb
    ├── Dataset_Raw/
    │   ├── pretrained_models/
    │   │   └── spkrec-ecapa-voxceleb/
    │   │       ├── classifier.ckpt
    │   │       ├── embedding_model.ckpt
    │   │       ├── hyperparams.yaml
    │   │       ├── label_encoder.ckpt
    │   │       └── mean_var_norm_emb.ckpt
    │   └── speechbrain_model/
    │       ├── classifier.ckpt
    │       ├── embedding_model.ckpt
    │       ├── hyperparams.yaml
    │       ├── label_encoder.ckpt
    │       └── mean_var_norm_emb.ckpt
    └── Real_Dataset_Embeddings/
        ├── demo_emb1.npy
        ├── demo_emb1_same_speaker.npy
        └── demo_emb2.npy

```

## Model Architecture

<img width="1024" height="942" alt="image" src="https://github.com/user-attachments/assets/21676771-6a28-400f-a319-97acf3d9fd2b" />


## Requirements

### System Requirements
- Python 3.7+
- Sufficient RAM for processing audio files.

### Dependencies

```bash
pip install torch torchaudio
pip install speechbrain
pip install librosa
pip install numpy pandas tqdm
