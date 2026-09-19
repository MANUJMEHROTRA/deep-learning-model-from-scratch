import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from config import get_config, get_weights_file_path, latest_weights_file_path
from torch.utils.data import DataLoader,Dataset,random_split
from dataset import BilingualDataset
from tqdm import tqdm
from model import build_transform

from torch.utils.tensorboard import SummaryWriter

config = get_config()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds():
        # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['seq_len'], config['lang_src'], config['lang_tgt'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['seq_len'], config['lang_src'], config['lang_tgt'])

    train_dataloader = DataLoader(train_ds,batch_size=8)
    val_dataloader = DataLoader(val_ds,batch_size=1)

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt



def get_model(input_vocab_size,output_vocab_size,):
    model = build_transform(input_vocab_size , output_vocab_size, input_seq_len=config["seq_len"], head=config["head"], output_seq_len=config["seq_len"],  N=6, ffn_size=2048, dropout=0.1, d_model = 768)
    return model

def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu")
    print(f"Using Device:{device}")


   
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_ds()
    model = get_model(tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size())
    

    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id("[SOS]"),label_smoothing=0.1)
    writer = SummaryWriter(config["experiment_name"])

    intial_epoch = 0
    global_step = 0

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    

    if model_filename:
        print(f"preloading fileName:{model_filename}")

        state = torch.load(model_filename)
        intial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        optimizer.load_state_dict(state["optimizer_state_dict"])
        model.load_state_dict(state['model_state_dict'])



    for epoch in range(intial_epoch,config["num_epochs"]):
        model.train()
        loss = 0
        batch_iterator = tqdm(train_dataloader,desc=f"Processing epoc: {epoch:02d}")
        for idx,batch in enumerate(train_dataloader):
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input,decoder_mask )
            project_output = model.project(decoder_output)

            loss = loss_fn(project_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train_loss",loss.item(),global_step)
            writer.flush()


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step +=1
        model_filename = get_weights_file_path(config,f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
            },model_filename)


if __name__=="__main__":

   train_model()


