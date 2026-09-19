import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,data,src_tokenizer,tgt_tokenizer,seq_len,src_lng="en",tgt_lng="it"):
        super().__init__()
        self.data = data
        self.src_lng =src_lng
        self.tgt_lng = tgt_lng
        self.seq_len = seq_len
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_sos = torch.tensor(self.src_tokenizer.token_to_id("[SOS]"), dtype=torch.int64).unsqueeze(0)
        self.src_eos = torch.tensor(self.src_tokenizer.token_to_id("[EOS]"), dtype=torch.int64).unsqueeze(0)
        self.src_pad = torch.tensor(self.src_tokenizer.token_to_id("[PAD]"), dtype=torch.int64).unsqueeze(0)
        self.tgt_sos = torch.tensor(self.tgt_tokenizer.token_to_id("[SOS]"), dtype=torch.int64).unsqueeze(0)
        self.tgt_eos = torch.tensor(self.tgt_tokenizer.token_to_id("[EOS]"), dtype=torch.int64).unsqueeze(0)
        self.tgt_pad = torch.tensor(self.tgt_tokenizer.token_to_id("[PAD]"), dtype=torch.int64).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_tgt_pair = self.data[index]
        source_sent = source_tgt_pair["translation"][self.src_lng]
        tgt_sent = source_tgt_pair["translation"][self.tgt_lng]

        enc_inp_token = self.src_tokenizer.encode(source_sent).ids
        dec_inp_sent = self.tgt_tokenizer.encode(tgt_sent).ids

        enc_num_padding_tokens = self.seq_len - len(enc_inp_token) - 2 
        dec_num_padding_tokens = self.seq_len - len(dec_inp_sent) - 1

        if enc_num_padding_tokens <0 or dec_num_padding_tokens<0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                     self.src_sos,
             torch.tensor(enc_inp_token, dtype=torch.int64),
                 self.src_eos,
                 torch.tensor([self.src_pad] * enc_num_padding_tokens, dtype=torch.int64)
             ]
        )

        decoder_input = torch.cat(
            [
                self.tgt_sos,
                torch.tensor(dec_inp_sent, dtype=torch.int64),
                torch.tensor([self.tgt_pad] * dec_num_padding_tokens, dtype=torch.int64)
             ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_inp_sent, dtype=torch.int64),
                self.tgt_eos,
                torch.tensor([self.tgt_pad] * dec_num_padding_tokens, dtype=torch.int64)
             ]
        )

        # assert eco

        return {
            "encoder_input": encoder_input, 
            "decoder_input":decoder_input, 
            "encoder_mask": (encoder_input != self.src_pad).unsqueeze(0).unsqueeze(0).int(),
            "decoder_mask": (decoder_input != self.tgt_pad).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": source_sent,
            "tgt_text": tgt_sent,

        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1).type(torch.int16)
    return mask==0

