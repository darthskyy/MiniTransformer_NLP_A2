"""
NLP Assignment 2: Transformer for character level modelling
adapted by Simbarashe Mawere from nanogpt
"""

# pylint: disable=C0103
# pylint: disable=C0301
import argparse
import math
import time
import logging
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F


# -------------------------------------------------------SETUP----------------------------------------------------------------
# arguments
parser = argparse.ArgumentParser(
    "Transformer for character level modelling adapted by Simbarashe Mawere from nanogpt"
)
parser.add_argument("--language", type=str, default="nr")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--seq_length", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=2)
parser.add_argument("--eval_every", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--example", action="store_true")
parser.add_argument("--fix_embed", action="store_true")
parser.add_argument("--scheduler", action="store_true")
parser.add_argument("--norm_after", action="store_true")
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--eval_iterations", type=int, default=500)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_file", type=str, default="x.log")
parser.add_argument("--save", type=str, default="model.pth")
args = parser.parse_args()

# hyperparameters
BATCH_SIZE = args.batch_size
SEQ_LENGTH = args.seq_length
NUM_EPOCHS = args.num_epochs
EVAL_EVERY = args.eval_every
LR = args.lr
DEVICE = args.device
EVAL_ITERATIONs = args.eval_iterations
EMBEDDING_DIM = args.embedding_dim
NUM_HEADS = args.num_heads
NUM_LAYERS = args.num_layers
DROPOUT = args.dropout
SEED = args.seed
LANGUAGE = args.language
NORM_BEFORE = False if args.norm_after else True
# ------------

# logging
# * setting up the logging
logger = logging.getLogger(f"train_{args.language}")
logger.setLevel("DEBUG")
formatter = logging.Formatter("%(message)s")  # Modified line

# setting up the file handler
fh = logging.FileHandler(f"logs/{args.log_file}")
fh.setLevel("DEBUG")

# setting up the console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel("DEBUG")

# setting up the formatter
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# adding the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
torch.manual_seed(SEED)


# ---------------------------------------------------DATA PROCESSING-------------------------------------------------------
# data reading in a plain text file and doing character level encoding since this is a character level modelling
with open(f"data/{LANGUAGE}.train", "r", encoding="utf-8") as f:
    text_train = f.read()

with open(f"data/{LANGUAGE}.valid", "r", encoding="utf-8") as f:
    text_valid = f.read()

with open(f"data/{LANGUAGE}.test", "r", encoding="utf-8") as f:
    text_test = f.read()

all_chars = sorted(list(set(text_train)))
vocab_size = len(all_chars)
# create a mapping from characters to integers
ch_to_idx = {ch: i for i, ch in enumerate(all_chars)}
idx_to_ch = {i: ch for i, ch in enumerate(all_chars)}


def encode(string_: str):
    return [ch_to_idx[ch] for ch in string_]


def decode(list_: list):
    return "".join([idx_to_ch[i] for i in list_])


# setting the data into pytorch tensors to make batching and sending to gpu easier
train_data = torch.tensor(encode(text_train), dtype=torch.long)
val_data = torch.tensor(encode(text_valid), dtype=torch.long)
test_data = torch.tensor(encode(text_test), dtype=torch.long)

dataset = {"train": train_data, "val": val_data, "test": test_data}

# normal batching
NUM_BATCHES = len(train_data) // BATCH_SIZE  # defining the number of batches per epoch
# NUM_BATCHES = 10000 # num_batches_per_epoch capped to 5000 to improve training time but otherwise the output is roughly the same
EVAL_EVERY = NUM_BATCHES // 10  # evaluating every 10th of the entire batch set

print(NUM_BATCHES)


def get_batch(dataset_in):
    # generate a small batch of data of inputs x and targets y via random sampling
    out = {}
    for split in dataset_in:
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - SEQ_LENGTH, (BATCH_SIZE,))
        x = torch.stack([data[i : i + SEQ_LENGTH] for i in ix])
        y = torch.stack([data[i + 1 : i + SEQ_LENGTH + 1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        out[split] = (x, y)
    return out


@torch.no_grad()  # disabling gradient decent for evaluation on the train and validation sets
def estimate_loss(dataset_in, test: bool = False):
    out = {}
    model.eval()
    splits = ["test"] if test else ["train", "val"]
    for split in splits:
        losses_est = torch.zeros(EVAL_ITERATIONs)
        for k in range(EVAL_ITERATIONs):
            X, Y = get_batch(dataset_in)[split]
            _, loss_est = model(
                X, Y
            )  # outputs the logits and the loss but no real need for the logits at this moment
            losses_est[k] = loss_est.item()
        out[split] = losses_est.mean()

    model.train()
    return out


# ---------------------------------------------------MODEL IMPLEMENTATION-------------------------------------------------------
# IMPLEMENTATION OF TRANSFORMER: AS AN AUTOREGRESSIVE DECODER ONLY ARCH
# The model will be assembled from the following components:
# 1. Token Embedding: a simple lookup table that maps integers to vectors
# 2. Positional Encoding: a fixed embedding that encodes the position of the token in the sequence
# 3. NBlock: a transformer block that contains a multi-headed attention layer and a feedforward layer
#  3.1. MultiHeadAttention: a module that contains multiple attention heads
#   3.1.1. AttentionHead: a single attention head that computes the attention from QKV matrices
#   3.1.2. FeedForward: a simple feedforward layer with a ReLU activation
#  3.2. FeedForward: a simple feedforward layer with a ReLU activation
# 4. MiniTransformer: a stack of NBlock layers with a final linear layer for output
#  4.1. generate: a method that generates new tokens from the model
# 5. Training loop: a simple training loop that samples batches of data and updates the model parameters
class AttentionHead(nn.Module):
    """
    A single head containing the query, key and value matrices sorely needed for attention to work.
    Chose to apply the tril mask (triangular zero matrix) to prevent the model peeking into the future. This is going to be decoder only so that makes sense.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(SEQ_LENGTH, SEQ_LENGTH)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        _, T, C = x.shape  # B: batch size, T: sequence length, C: embedding dimension
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)

        # computing the attention from QKV
        attention = q @ k.transpose(-2, -1) * C**-0.5

        # applying to mask to obscure future tokens/characters
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # softmax for the probabilities and then adding a bit of dropout if needed
        attention = F.softmax(attention, dim=-1)  # (B, T, T)
        attention = self.dropout(attention)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)

        return (
            attention @ v
        )  # once we get the attention, we multiply it with the value matrix to get the results


class MultiHeadAttention(nn.Module):
    """WELCOME TO CERBERUS: THE MULTIHEADED ATTENTION MATRIX"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])

        # simple linear projection for the output after the attention has been applied
        # and dropout for regularisation
        self.proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """simple ffl with regulisation. we expand it to get more features their 'compress' to get back regular matrix size"""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class NBlock(nn.Module):
    """
    Transformer block (a decoder only)
    Can perform normalisation before mha and ffl or after depending on configuration.
    """

    def __init__(self, embedding_dim, num_heads, norm_before: bool = True):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = embedding_dim // num_heads
        self.norm_before = norm_before
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.ffl = FeedFoward(embedding_dim)

        # layers for add + norm operation
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # we perform the simple decoder actions except we do norm + add
        if self.norm_before:
            x = x + self.mha(self.layer_norm1(x))
            x = x + self.ffl(self.layer_norm2(x))
        else:
            x = self.layer_norm1(x + self.mha(x))
            x = self.layer_norm2(x + self.ffl(x))
        return x


# super simple bigram model
class MiniTransformer(nn.Module):

    def __init__(self, fix_embed: bool = False, norm_before: bool = True):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.fix_embed = fix_embed
        self.norm_before = norm_before
        if self.fix_embed:
            self.position_embedding_table = self._get_positional_embeddings()
            print("running with fixed positional embeddings")
        else:
            self.position_embedding_table = nn.Embedding(SEQ_LENGTH, EMBEDDING_DIM)
        self.blocks = nn.Sequential(
            *[
                NBlock(EMBEDDING_DIM, num_heads=NUM_HEADS, norm_before=norm_before)
                for _ in range(NUM_LAYERS)
            ]
        )
        self.layer_norm_f = nn.LayerNorm(EMBEDDING_DIM)  # final layer norm
        self.linear_out = nn.Linear(EMBEDDING_DIM, vocab_size)

    def _get_positional_embeddings(self):
        position_enc = torch.zeros(SEQ_LENGTH, EMBEDDING_DIM)
        position = torch.arange(0, SEQ_LENGTH, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, EMBEDDING_DIM, 2).float()
            * (-math.log(10000.0) / EMBEDDING_DIM)
        )
        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)
        # we freeze the embeddings so that they do not get changed in training
        return nn.Embedding.from_pretrained(position_enc, freeze=True)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layer_norm_f(x)  # (B,T,C)
        logits_forward = self.linear_out(x)  # (B,T,vocab_size)

        if targets is None:
            loss_forward = None
        else:
            B, T, C = logits_forward.shape
            logits_forward = logits_forward.view(B * T, C)
            targets = targets.view(B * T)
            loss_forward = F.cross_entropy(logits_forward, targets)

        return logits_forward, loss_forward

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -SEQ_LENGTH:]
            # get the predictions
            logits_pred, _ = self(idx_cond)
            # focus only on the last time step
            logits_pred = logits_pred[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits_pred, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# ---------------------------------------------------MODEL TRAINING-------------------------------------------------------
model = MiniTransformer(fix_embed=args.fix_embed, norm_before=NORM_BEFORE)
m = model.to(DEVICE)
# print(m)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# create a learning rate scheduler
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    SCHEDULE_EVERY_STEPS = (NUM_BATCHES // 4) * 3  # schedule every 3/4 of the batches

# training loop
for epoch in range(1, NUM_EPOCHS + 1):
    print("START TRAINING")
    # perform batch training
    EPOCH_START = time.time()
    for batch_id in range(1, NUM_BATCHES):
        if batch_id == 1:
            batch_start = time.time()
        # every once in a while evaluate the loss on train and val sets
        if batch_id % EVAL_EVERY == 0 or batch_id == NUM_BATCHES - 1:
            batch_time = ((time.time() - batch_start) * 1000) / EVAL_EVERY
            losses = estimate_loss(dataset)
            bpc_train = losses["train"] / math.log(2, 10)
            bpc_val = losses["val"] / math.log(2, 10)
            logger.info(
                f"| EPOCH {epoch} | {batch_id}/{NUM_BATCHES} batches | lr {optimizer.param_groups[0]['lr']} | ms/batch {batch_time:.2f} | train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | train bpc {bpc_train:.2f} | val bpc {bpc_val:.2f} "
            )
            batch_start = time.time()

        # sample a batch of data
        xb, yb = get_batch(dataset)["train"]

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # update the learning rate
        if args.scheduler and batch_id % SCHEDULE_EVERY_STEPS == 0:
            scheduler.step()

    epoch_time = time.time() - EPOCH_START
    logger.info(f"| end of epoch {epoch} | time: {epoch_time:.2f}s")


# ---------------------------------------------------MODEL TEST EVAL-------------------------------------------------------
# evaluate on the test set
losses = estimate_loss(dataset, test=True)
bpc = losses["test"] / math.log(2, 10)
logger.info(f"| end of training | test loss {losses['test']:.4f} | test bpc {bpc:.2f}")


# ---------------------------------------------------MODEL SAVING & PRED---------------------------------------------------
# save the model
if args.save:
    torch.save(model.state_dict(), f"models/{args.save}")


if args.example:
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
