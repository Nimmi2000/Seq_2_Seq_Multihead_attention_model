
# Seq2Seq Multihead Attention Model

This repository provides an implementation of a Sequence-to-Sequence (Seq2Seq) Transformer model with multi-head attention for neural machine translation using PyTorch.

## Model Overview
- **Architecture:** Transformer-based encoder-decoder with multi-head self-attention
- **Framework:** PyTorch
- **Tokenization:** Word-level tokenizers (with [SOS], [EOS], [PAD], [UNK] tokens)
- **Dataset:** [opus_books](https://huggingface.co/datasets/opus_books) (default: English to Dutch)
- **Features:**
	- Customizable sequence length, batch size, and model dimensions
	- Training and validation split
	- Model checkpointing
	- TensorBoard logging

## Project Structure
- `training.py` — Main script for training the model
- `source_model.py` — Transformer model architecture
- `translation_data.py` — Dataset and preprocessing utilities
- `config.py` — Configuration and hyperparameters
- `tokenizer_en.json`, `tokenizer_nl.json` — Saved tokenizers

## Requirements
- Python 3.8+
- PyTorch
- HuggingFace Datasets
- Tokenizers
- tqdm
- tensorboard

Install dependencies with:
```sh
pip install torch datasets tokenizers tqdm tensorboard
```

## Training the Model
To start training, run the following command in your terminal:

```sh
python training.py
```

This will use the default configuration in `config.py` and begin training the model from scratch. Model checkpoints will be saved in the `trained_weights` directory.

## Inference Example
To perform inference (translation) after training, you can load the trained model and tokenizers, then use the model's `encode`, `decode`, and `projection` methods. Example code snippet:

```python
import torch
from source_model import build_transformer
from config import get_config, get_weights_file_path
from tokenizers import Tokenizer

config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizers
src_tokenizer = Tokenizer.from_file('tokenizer_en.json')
tgt_tokenizer = Tokenizer.from_file('tokenizer_nl.json')

# Build and load model
model = build_transformer(
		src_tokenizer.get_vocab_size(),
		tgt_tokenizer.get_vocab_size(),
		config['sequence_length'],
		config['sequence_length'],
		config['d_model']
).to(device)

checkpoint = torch.load(get_weights_file_path(config, '<epoch_number>'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input (replace with your sentence)
input_sentence = "This is a test."
input_ids = src_tokenizer.encode(input_sentence).ids
# ... (pad and add [SOS]/[EOS] as in training)
# ... (create encoder_input, encoder_mask, decoder_input, decoder_mask)
# ... (run model.encode, model.decode, model.projection)
```

> **Note:** For a full inference pipeline, adapt the preprocessing and decoding steps from `translation_data.py` and `training.py`.

## License
See [LICENSE](LICENSE).
