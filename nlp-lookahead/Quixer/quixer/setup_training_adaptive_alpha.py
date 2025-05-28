import random
import os
import time
import math
from tqdm import tqdm
from pathlib import Path
from typing import Any, Optional, Tuple, Callable
import torch
from torch.types import Device
from quixer.quixer_model import Quixer
from quixer.baseline_models import Transformer, LSTM, FNet

import numpy as np

from torch.nn.modules.loss import _Loss
import torchtext
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt

from torch.types import Device
# =============================================================================
# Adaptive Lookahead Optimizer (using adaptive α as in your first code block)
# =============================================================================
class AdaptiveLookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha_min=0.1, alpha_max=0.9, k=5):
        if not 0.0 <= alpha_min <= alpha_max <= 1.0:
            raise ValueError(f"Invalid alpha range: {alpha_min} to {alpha_max}")
        if k < 1:
            raise ValueError(f"Invalid k: {k}")
        
        self.base_optimizer = base_optimizer
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.k = k

        # Count fast (base) steps
        self._step_count = 0
        
        # For adaptive alpha calculation
        self.best_val_loss = float('inf')
        self.current_alpha = (alpha_min + alpha_max) / 2  # start at the midpoint
        
        # Sigmoid parameters for controlling the mapping
        self.sigmoid_scale = 5.0   # controls steepness
        self.sigmoid_shift = 0.0   # controls the center
        self.damping = 0.9         # damping factor for smooth updates

        self.alpha_history = []

        # Create a copy of the fast weights for the slow (lookahead) update.
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = []
            for p in group['params']:
                sp.append(p.clone().detach())
            self.slow_params.append(sp)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-self.sigmoid_scale * (x - self.sigmoid_shift)))

    def update_alpha(self, val_loss):
        eps = 1e-8
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Compute ratio: a ratio below 1 means current loss is worse than best.
        ratio = self.best_val_loss / (val_loss + eps)

        # Map ratio to [0, 1] using a sigmoid centered at 0.5.
        normalized = 1.0 / (1.0 + np.exp(-self.sigmoid_scale * (ratio - 0.5)))
        new_alpha = self.alpha_min + normalized * (self.alpha_max - self.alpha_min)
        
        # Use damping to smooth updates.
        self.current_alpha = self.damping * self.current_alpha + (1 - self.damping) * new_alpha
        self.current_alpha = np.clip(self.current_alpha, self.alpha_min, self.alpha_max)
        self.alpha_history.append(self.current_alpha)
        # Diagnostic printout (optional)
        print(f"\n[Adaptive Lookahead] Val Loss: {val_loss:.6f}, Ratio: {ratio:.6f}, Best: {self.best_val_loss:.6f}, Alpha: {self.current_alpha:.4f}")
        return self.current_alpha

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None, val_loss: Optional[float] = None):
        # Perform fast (base optimizer) step.
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        # Update adaptive alpha if a validation loss has been provided.
        if val_loss is not None:
            self.update_alpha(val_loss)

        # Every k fast steps, perform the slow update.
        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    slow += self.current_alpha * (p.data - slow)
                    p.data.copy_(slow)

        return loss

# =============================================================================
# Data Preparation (same as your original functions)
# =============================================================================
def batchify_s2s(
    data: torch.Tensor,
    batch_size: int,
    window_size: int,
    pad_token_id: int,
    device: Device,
) -> torch.Tensor:
    """
    Takes in a sequence of token IDs as a torch tensor `data` and returns a torch tensor containing
    the training data with shape `[number of batches + window_size, batch_size]`.

    Each batch is represented by `window_size` contiguous rows in the returned tensor and
    can be extracted using the `get_batch_s2s` function.

    A sequence of pad tokens of length `window_size-1` is prepended to the data so as to
    provide a context window for the first token.

    Args:
      data: A 1D torch tensor containing a sequence of token IDs.
      batch_size: The number of sequences each batch should have.
      window_size: How many tokens are considered in each context window (each of which is a sequence in the batch).
      pad_token_id: The ID of the pad token, as supplied by the tokenizer.
      device: Torch device the returned tensor is to be created on.

    Returns:
      Tensor containing data for each batch prepared for a next token prediction language
      modelling task.
    """
    batch_nr_of_elements = batch_size * window_size
    nr_of_batches = (data.size(0) - 1) // batch_nr_of_elements

    # Discard tokens at the end of the data that do not fill a whole batch
    batched_data = (
        data[: nr_of_batches * batch_nr_of_elements]
        .view(batch_nr_of_elements, nr_of_batches)
        .T
    )

    # Data for the first batch
    window_data = torch.cat(
        (
            # Adds a sequence of pad tokens of length `window_size-1`
            # to provide a context window for the first token.
            torch.full((window_size, 1), pad_token_id, device=device),
            # Context for the first row of tokens in `batched_data`
            batched_data[-window_size:, :-1],
        ),
        dim=1,
    )

    return torch.cat((window_data, batched_data))

def get_batch_s2s(
    source: torch.Tensor, i: int, window_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the `i`th batch; expects one of the tensors returned by `setup_dataset`.

    Args:
      source: Tensor containing data.
      i: Index of the batch.
      window_size: Context window size.
    Returns:
      The `i`th batch.
    """
    return source[i : i + window_size].T, source[i + window_size]

def initialise_weights(model: torch.nn.Module) -> None:
    """
    Initialises model weights.
    """

    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(_init_weights)


def setup_dataset(
    device: Device, batch_size: int, window_size: int
) -> Tuple[torchtext.vocab.Vocab, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]:
    """
    Downloads and tokenizes the Penn TreeBank dataset, and then sets it up for a
    next-word prediction task.

    Args:
      device: Device to store dataset on.
      batch_size: Size of the batches.
      window_size: Size of the context window.

    Returns:
      Vocabulary represented by a torchtext.vocab.Vocab instance along with
      three torch tensors containing the training, validation and test data.
    """

    # Download dataset from the Hugging Face Hub / load dataset
    raw_dset = load_dataset("ptb_text_only")

    # Get training data in PyArrow format
    train_iter = raw_dset["train"].data[0]
    # Convert from arrow array to native Python list
    train_iter = [s.as_py() for s in train_iter]

    # Get torchtext tokenizer
    tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter), specials=["<pad>", "<unk>", "<eos>"]
    )
    # Define unknown word as the default index to use
    vocab.set_default_index(vocab["<unk>"])

    def data_process(raw_text_iter) -> torch.Tensor:
        """
        Converts raw text into a flat Tensor of token indices.
        """
        data = [
            torch.tensor(vocab(tokenizer(item)) + [vocab["eos"]], dtype=torch.long)
            for item in raw_text_iter
        ]
        return torch.cat(tuple(filter(lambda t: t.numel() > 1, data))).to(device)

    # Convert from arrow arrays to native Python lists
    train_sents = [s.as_py() for s in raw_dset["train"].data[0]]
    val_sents = [s.as_py() for s in raw_dset["validation"].data[0]]
    test_sents = [s.as_py() for s in raw_dset["test"].data[0]]

    # Flatten datasets into one long tokenised string each
    train_flat = data_process(train_sents)
    val_flat = data_process(val_sents)
    test_flat = data_process(test_sents)

    # Get padding token
    PAD_TOKEN = vocab["<pad>"]

    # Prepare data for a next-token prediction language modelling task
    train_iter = batchify_s2s(train_flat, batch_size, window_size, PAD_TOKEN, device)
    val_iter = batchify_s2s(val_flat, batch_size, window_size, PAD_TOKEN, device)
    test_iter = batchify_s2s(test_flat, batch_size, window_size, PAD_TOKEN, device)

    return vocab, (train_iter, val_iter, test_iter), PAD_TOKEN

# =============================================================================
# Model creation functions (left unchanged, except for your own model choices)
# =============================================================================

def create_model(
    hyperparams: dict[str, Any], device: Device, vocabulary_size: int
) -> torch.nn.Module:
    """
    Selects and creates model based on hyperparameters passed.

    Args:
      hyperparams: Model hyperparameters.
      device: Device the model will be run on.
      vocabulary_size: Size of the vocabulary.
    Returns:
      An instance of a torch model based on the hyperparameters passed.
    """
    model_str = hyperparams["model"]
    model: torch.nn.Module
    if model_str == "Quixer":
        model = Quixer(
            n_qubits=hyperparams["qubits"],
            n_tokens=hyperparams["window"],
            qsvt_polynomial_degree=hyperparams["layers"],
            n_ansatz_layers=hyperparams["ansatz_layers"],
            vocabulary_size=vocabulary_size,
            embedding_dimension=hyperparams["dimension"],
            dropout=hyperparams["dropout"],
            batch_size=hyperparams["batch_size"],
            device=device,
        )
    elif model_str == "FNet":
        model = FNet(
            vocab_size=vocabulary_size,
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            dropout=hyperparams["dropout"],
        )
    elif model_str == "Transformer":
        model = Transformer(
            emb_dim=hyperparams["dimension"],
            hid_dim=4 * hyperparams["dimension"],
            n_heads=hyperparams["heads"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
        )
    elif model_str == "LSTM":
        model = LSTM(
            emb_dim=hyperparams["dimension"],
            hid_dim=hyperparams["dimension"],
            n_layers=hyperparams["layers"],
            vocab_size=vocabulary_size,
            dropout=hyperparams["dropout"],
        )
    else:
        raise ValueError(f"Unrecognized model: {model_str}")

    return model


# =============================================================================
# Updated training loop using adaptive alpha (with separate 5% adaptation data)
# =============================================================================
def train_epoch(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    adapt_data: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: _Loss,
    clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    window_size: int,
):
    """
    Runs training loop for one epoch.
    """
    model.train()
    epoch_loss = 0
    n_train_batches = train_data.shape[0] - window_size

    # Prepare adaptation data indices (for the held-out 5%)
    n_adapt_batches = adapt_data.shape[0] - window_size
    adapt_idxs = list(range(n_adapt_batches))
    random.shuffle(adapt_idxs)
    adapt_counter = 0

    # Shuffle the training batch order
    train_idxs = list(range(n_train_batches))
    random.shuffle(train_idxs)

    for batch_idx in tqdm(train_idxs, total=n_train_batches, desc="Training"):
        # Get a training batch (95% split)
        x, y = get_batch_s2s(train_data, batch_idx, window_size)
        optimizer.zero_grad()
        yhat, _ = model(x)
        loss = loss_function(yhat, y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # --- Adaptive alpha update using a mini-batch from held-out (5%) data ---
        # Cycle through the adaptation indices; reshuffle if exhausted.
        if adapt_counter >= len(adapt_idxs):
            random.shuffle(adapt_idxs)
            adapt_counter = 0
        adapt_batch_idx = adapt_idxs[adapt_counter]
        adapt_counter += 1

        # Compute loss on adaptation mini-batch in evaluation mode.
        model.eval()
        with torch.no_grad():
            adapt_x, adapt_y = get_batch_s2s(adapt_data, adapt_batch_idx, window_size)
            adapt_yhat, _ = model(adapt_x)
            adapt_loss = loss_function(adapt_yhat, adapt_y)
        model.train()

        # Call optimizer step with the adaptation loss for updating α.
        optimizer.step(val_loss=adapt_loss.item())
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / n_train_batches

def evaluate(model: torch.nn.Module, data: torch.Tensor, loss_function: _Loss, window_size: int) -> float:
    model.eval()
    epoch_loss = 0
    n_batches = data.shape[0] - window_size
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
            x, y = get_batch_s2s(data, batch_idx, window_size)
            yhat, _ = model(x)
            loss = loss_function(yhat, y)
            epoch_loss += loss.item()
    return epoch_loss / n_batches

def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    elapsed_time = end_time - start_time
    return int(elapsed_time // 60), int(elapsed_time % 60)

def train_cycle(model: torch.nn.Module, hyperparams: dict[str, Any],
                train_iter: torch.Tensor, val_iter: torch.Tensor,
                test_iter: torch.Tensor) -> float:
    folder_path = Path("./trained_models")
    folder_path.mkdir(exist_ok=True, parents=True)
    checkpoint_fpath = folder_path / f"q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"

    # -------------------------------------------------------------------------
    # Split the batchified training data into:
    #   - 95% for training (fast updates)
    #   - 5% for computing the adaptation (alpha) loss.
    # -------------------------------------------------------------------------
    n_train_rows = train_iter.shape[0]
    split_idx = int(0.95 * n_train_rows)
    training_data = train_iter[:split_idx]
    adapt_data = train_iter[split_idx:]

    # Set up the optimizer: first the base optimizer, then wrap with AdaptiveLookahead.
    base_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )
    optimizer = AdaptiveLookahead(
        base_optimizer=base_optimizer,
        alpha_min=hyperparams.get("alpha_min", 0.1),
        alpha_max=hyperparams.get("alpha_max", 0.9),
        k=hyperparams.get("lookahead_k", 10),
    )
    # Optionally, you can set sigmoid parameters:
    optimizer.sigmoid_scale = hyperparams.get("sigmoid_scale", 5.0)
    optimizer.sigmoid_shift = hyperparams.get("sigmoid_shift", 0.0)

    # Set up learning rate scheduler if required.
    scheduler = None
    if hyperparams.get("lr_sched", None) == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_optimizer, T_0=hyperparams["restart_epochs"]
        )

    loss_function = torch.nn.CrossEntropyLoss()

    def _evaluate(data_tensor: torch.Tensor) -> float:
        return evaluate(model, data_tensor, loss_function, hyperparams["window"])

    best_valid_loss = float("inf")
    train_loss_history = []
    val_loss_history = []
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(model, training_data, adapt_data, optimizer, loss_function,
                                 hyperparams["max_grad_norm"], scheduler, hyperparams["window"])
        valid_loss = _evaluate(val_iter)
        train_loss_history.append(train_loss)
        val_loss_history.append(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train ppl: {math.exp(train_loss):.3f}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss):.3f}")

    # Plot the training and validation loss curves.
    plt.figure()
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    loss_plot_path = checkpoint_fpath.parent / f"{checkpoint_fpath.stem}_loss.png"
    plt.savefig(loss_plot_path)
    plt.show()

    # Plot the adaptive α history over iterations.
    plt.figure()
    plt.plot(optimizer.alpha_history, label="Alpha Values")
    plt.xlabel("Iteration")
    plt.ylabel("Alpha")
    plt.title("Adaptive Alpha History")
    plt.legend()
    alpha_plot_path = checkpoint_fpath.parent / f"{checkpoint_fpath.stem}_alpha.png"
    plt.savefig(alpha_plot_path)
    plt.show()

    model.load_state_dict(torch.load(checkpoint_fpath))
    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss):.3f}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss):.3f}")

    return test_loss

# =============================================================================
# Utility: seed, get_train_evaluate, etc.
# =============================================================================
def seed_everything(SEED: int) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def get_train_evaluate(device: torch.device) -> Callable:
    def train_evaluate(parameterization: dict[str, Any]) -> float:
        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")
        seed_everything(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"]
        )
        model = create_model(parameterization, device, len(vocab))
        initialise_weights(model)
        model = model.to(device)
        valid_loss = train_cycle(model, parameterization, train_iter, val_iter, test_iter)
        return valid_loss
    return train_evaluate
