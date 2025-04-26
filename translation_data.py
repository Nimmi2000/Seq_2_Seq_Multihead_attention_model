import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TranslationData(Dataset):

    def __init__(self, ds, tokenizer_source, tokenizer_target, source_language, target_language, sequence_length) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.source_language = source_language
        self.target_language = target_language
        self.sequence_length = sequence_length

        self.sos_token = torch.tensor([tokenizer_source.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_source.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_source.token_to_id('[PAD]')], dtype = torch.int64)

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, index):
        source_target_pair = self.ds[index]

        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        encode_input_tokens = self.tokenizer_source.encode(source_text).ids
        decode_input_tokens = self.tokenizer_target.encode(target_text).ids

        encode_num_padding_tokens = self.sequence_length - len(encode_input_tokens) - 2
        decode_num_padding_tokens = self.sequence_length - len(decode_input_tokens) - 1

        if encode_num_padding_tokens < 0 or decode_num_padding_tokens < 0:
            return ValueError('Negative padding found!!!')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encode_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encode_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decode_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * decode_num_padding_tokens, dtype=torch.int64)
            ]
        )

        labels = torch.cat(
            [
                torch.tensor(decode_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decode_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.sequence_length
        assert decoder_input.size(0) == self.sequence_length
        assert labels.size(0) == self.sequence_length

        return {"encoder_input": encoder_input ,"decoder_input": decoder_input, "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), "label": labels, "source_text" : source_text, "target_text" : target_text}
    

def causal_mask(size):

    mask = torch.triu(torch.ones(1,size,size), diagonal = 1).type(torch.int)

    return mask == 0
