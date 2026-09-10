# Train a small character-level GPT on tiny-Shakespeare.
# Override values that train.py defines as globals (see configurator.py).

out_dir = 'out-shakespeare-char'
eval_interval = 250   # evaluate fairly often since runs are short
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context length: how many characters of history the model sees

# a baby GPT (~10.7M params)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3   # small network -> can use a higher LR
max_iters = 5000
lr_decay_iters = 5000  # usually ~= max_iters
min_lr = 1e-4          # ~= learning_rate / 10
beta2 = 0.99           # slightly higher because each iter sees few tokens

warmup_iters = 100
