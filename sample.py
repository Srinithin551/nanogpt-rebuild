"""
Sample from a trained model.

From a checkpoint you trained:
    python sample.py --out_dir=out-shakespeare-char
From an OpenAI GPT-2 model:
    python sample.py --init_from=gpt2-xl --start="Hello, I am" --num_samples=3

Config is handled by configurator.py (--key=value overrides).
"""
import os
import pickle
from contextlib import nullcontext
import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'          # 'resume' (from out_dir) or a 'gpt2*' name
out_dir = 'out'               # where to find ckpt.pt when init_from == 'resume'
start = "\n"                  # prompt; or "FILE:prompt.txt" to read from a file
num_samples = 10              # how many independent samples to generate
max_new_tokens = 500          # tokens generated per sample
temperature = 0.8             # <1 = more predictable, >1 = more random/diverse
top_k = 200                   # keep only the top_k most likely tokens per step
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())  # overrides from CLI
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' \
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# --- load the model ---
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'  # strip torch.compile's prefix if present
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# --- set up the tokenizer (encode/decode) ---
# Prefer the dataset's meta.pkl (character-level). Fall back to GPT-2 BPE.
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 BPE encodings...")
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# --- encode the prompt ---
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # (1, t)

# --- generate ---
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
