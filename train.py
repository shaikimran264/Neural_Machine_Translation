import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from indicnlp.tokenize import indic_tokenize
import string
import spacy
nlp = spacy.load("en_core_web_sm")
import pickle
from indicnlp.tokenize import indic_tokenize
import fasttext.util
import fasttext
import math
from torch.utils.data import DataLoader, Dataset
# Download the Hindi word vectors
# fasttext.util.download_model('hi', if_exists='ignore')

from tqdm import tqdm
import torch.optim as optim
from torch.amp import autocast, GradScaler


import pickle
with open('/content/drive/MyDrive/final_test_df','rb') as f:
  test_data=pickle.load(f)



device='cuda' if torch.cuda.is_available() else 'cpu'


source_sentences_test=[]
target_sentences_test=[]
id_val=[]

for language_pair, language_data in test_data.items():
    if(language_pair == "English-Hindi"):
      print(f"Language Pair: {language_pair}")
      for data_type, data_entries in language_data.items():
          print(f"  Data Type: {data_type}")
          for entry_id, entry_data in data_entries.items():
              source = entry_data["source"]
              target = entry_data["target"]
              if (data_type == "Test"):
                source_sentences_test.append(source)
                target_sentences_test.append(target)
                id_val.append(entry_id)
              else:
                source_sentences_train.append(source)
                target_sentences_train.append(target)
                id_train.append(entry_id)



test_x={'English':source_sentences_test,'Hindi':target_sentences_test}

test_df=pd.DataFrame(test_x)




with open('english_train_tokens','rb') as f:
  english_tokens_1=pickle.load(f)

with open('english_test_tokens','rb') as f:
  english_test_1=pickle.load(f)



with open('hindi_train_tokens','rb') as f:
  hindi_tokens_1=pickle.load(f)

en_train=english_tokens_1
en_test=english_test_1
hi_train=hindi_tokens_1



def encode_and_pad_fixed(sentences, word2index, max_length, device):

    encoded_tensors = []

    for sent in sentences:      
        encoded = [word2index["<SOS>"]] + \
                  [word2index.get(word, word2index["<UNK>"]) for word in sent] + \
                  [word2index["<EOS>"]]
      
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded += [word2index["<PAD>"]] * (max_length - len(encoded))

        encoded_tensors.append(torch.tensor(encoded, dtype=torch.long))    
    padded = torch.stack(encoded_tensors)
    return padded.to(device)

max_len = 50  # manually chosen
en_test_tensor = encode_and_pad_fixed(english_test_1, en_word2index, max_len, device)


print("English test shape:", en_test_tensor.shape)



def load_glove_embeddings(glove_path):
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index

glove_path = '/glove.6B.300d.txt'
glove_embeddings = load_glove_embeddings(glove_path)


import torch

from tqdm import tqdm
import json


class TestDataset(Dataset):
    def __init__(self, src_data):
        self.src_data = src_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long)


def collate_fn_test(batch, pad_id):
    max_len = max(len(x) for x in batch)
    padded = [torch.cat([x, torch.full((max_len - len(x),), pad_id)]) for x in batch]
    return torch.stack(padded, dim=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pad_id_src = en_word2index["<PAD>"]
pad_id_tgt = hi_word2index["<PAD>"]

model = TransformerMT(
    src_vocab_size=len(en_word2index),
    tgt_vocab_size=len(hi_word2index),
    src_embeddings=src_embeddings,
    tgt_embeddings=tgt_embeddings,
    src_pad_id=pad_id_src,
    tgt_pad_id=pad_id_tgt
).to(device)

checkpoint_path = "frf_transformer_epoch10.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Loaded checkpoint: {checkpoint_path}")


def greedy_decode(model, src, max_len=60, pad_id=0, device="cpu"):
    model.eval()
    batch_size = src.size(0)

    
    sos_id = 1
    tgt = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            logits = model(src, tgt)   # (B, T, vocab_size)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
            tgt = torch.cat([tgt, next_token], dim=1)

    return tgt  # (B, max_len)


test_dataset = TestDataset(en_test_encoded)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda b: collate_fn_test(b, pad_id_src)
)

save_path = "/content/drive/MyDrive/nnn_est_predictions.json"

all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Decoding", unit="batch"):
        src_ids = batch.to(device)

        
        batch_preds = greedy_decode(
            model, src_ids, max_len=60, pad_id=pad_id_tgt, device=device
        )

        
        batch_preds = [seq.tolist() for seq in batch_preds]
        all_preds.extend(batch_preds)


with open(save_path, "w", encoding="utf-8") as f:
    json.dump(all_preds, f, ensure_ascii=False, indent=2)

print(f"Saved test predictions to {save_path}")


import json
with open("nn_est_predictions.json", "r", encoding="utf-8") as f:
    preds = json.load(f)


import json
from sacrebleu.metrics import BLEU, CHRF
from rouge import Rouge  


decoded_refs = [
    r[0] if isinstance(r, list) else r
    for r in decoded_refs
]


refs_nested = [decoded_refs]


bleu_metric = BLEU()
chrf_metric = CHRF()

bleu = bleu_metric.corpus_score(decoded_preds, refs_nested)
chrf = chrf_metric.corpus_score(decoded_preds, refs_nested)


rouge = Rouge()
scores = rouge.get_scores(decoded_preds, decoded_refs, avg=True)


metrics = {
    "BLEU": bleu.score,  
    "chrF": chrf.score,  
    "ROUGE-1": scores["rouge-1"]["f"] * 100,
    "ROUGE-2": scores["rouge-2"]["f"] * 100,
    "ROUGE-L": scores["rouge-l"]["f"] * 100
}

print(metrics)



