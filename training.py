import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from torch.utils.tensorboard import SummaryWriter

from translation_data import TranslationData, causal_mask
from source_model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

def get_training_data(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def build_tokenizer(config , dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[SOS]', '[EOS]', '[PAD]', '[UNK]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_training_data(dataset, language), trainer = trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_dataset(config):

    dataset_raw = load_dataset('opus_books', f"{config['language_source']}-{config['language_target']}", split = "train")

    # Tokenizer
    tokenizer_source = build_tokenizer(config , dataset_raw, config['language_source'])
    tokenizer_target = build_tokenizer(config , dataset_raw, config['language_target'])

    # Split data for training and validation

    train_dataset_size = int(0.9*len(dataset_raw))
    validation_dataset_size = len(dataset_raw) - train_dataset_size
    
    train_data_raw , validation_data_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size])

    train_dataset = TranslationData(train_data_raw, tokenizer_source, tokenizer_target, config["language_source"], config["language_target"], config["sequence_length"])
    validation_dataset = TranslationData(validation_data_raw, tokenizer_source, tokenizer_target, config["language_source"], config["language_target"], config["sequence_length"])

    max_length_source = 0
    max_length_target = 0

    for item in dataset_raw:
        source_ids = tokenizer_source.encode(item["translation"][config["language_source"]]).ids
        target_ids = tokenizer_source.encode(item["translation"][config["language_target"]]).ids
        max_length_source = max(max_length_source, len(source_ids))
        max_length_target = max(max_length_target, len(target_ids))

    train_dataloader = DataLoader(train_dataset, batch_size = config["batch_size"], shuffle= True)
    validation_dataloader = DataLoader(validation_dataset, batch_size= 1, shuffle= True)

    return train_dataloader, validation_dataloader, tokenizer_source, tokenizer_target

def get_model(config, vocabulary_source_length, vocabulary_target_length):

    model = build_transformer(vocabulary_source_length, vocabulary_target_length, config["sequence_length"], config["sequence_length"], config["d_model"])
    return model

def train_model(config):

    #Defining device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader , validation_dataloader, tokenizer_source, tokenizer_target = get_dataset(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config["experiment_name"])

    #Optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"pre-loading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state["global_step"]

    loss_function = nn.CrossEntropyLoss(ignore_index= tokenizer_source.token_to_id('[PAD]'), label_smoothing= 0.1).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device) # Size : (B , 1, 1, Sequence Length)
            decoder_mask = batch["decoder_mask"].to(device) # Size : (B , 1, Sequence Length, Sequence Length)

            # Using the tensors
            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, sequence length, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, sequence length, d_model)

            projection_output = model.projection(decoder_output) # Batch, sequence length, target vocabulary size

            label = batch['label'].to(device) # Batch, sequence length

            # Batch, sequence length, target vocabulary size -> B * sequence length , target vocabulary size
            loss = loss_function(projection_output.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({"loss" : f"{loss.item():6.3f}"})

            #Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
        
        # Saving the model

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict(), 'global_step' : global_step}, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)