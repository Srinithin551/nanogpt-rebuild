"""
Prepare the tiny-Shakespeare dataset for CHARACTER-LEVEL language modeling.

Instead of using GPT-2's BPE tokenizer, we build the simplest possible
tokenizer: every unique character in the text becomes one integer. The whole
"vocabulary" is just the set of distinct characters (~65 of them).

Outputs written next to this file:
  - train.bin : the first 90% of the text as a stream of uint16 token ids
  - val.bin   : the remaining 10%, same format
  - meta.pkl  : vocab_size + the char<->int mappings, so we can decode later
"""
import os
import pickle
import numpy as np

# --- 1. Download the raw text (once) ----------------------------------------
# We save it right next to this script so re-running is a no-op after the first
# time. os.path.dirname(__file__) = the folder this .py file lives in.
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    import requests  # only needed for the one-time download
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# --- 2. Build the vocabulary ------------------------------------------------
# The vocabulary is just every distinct character, sorted for a stable order.
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# --- 3. Build the tokenizer (char <-> int) ----------------------------------
stoi = {ch: i for i, ch in enumerate(chars)}  # string-to-int
itos = {i: ch for i, ch in enumerate(chars)}  # int-to-string

def encode(s):
    return [stoi[c] for c in s]        # "hi" -> [46, 47]

def decode(l):
    return ''.join(itos[i] for i in l) # [46, 47] -> "hi"

# --- 4. Train / validation split --------------------------------------------
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# --- 5. Encode both splits to integer ids -----------------------------------
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# --- 6. Save to .bin as a flat uint16 array ---------------------------------
# uint16 holds 0..65535, plenty for a 65-char vocab. Saving as raw bytes lets
# train.py memory-map the file later without loading it all into RAM.
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# --- 7. Save the meta info so we can decode later ---------------------------
# train.py reads vocab_size from here; sample.py reads stoi/itos to turn a
# prompt into ids and the model's output ids back into text.
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
