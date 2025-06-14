import random
import os
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np

import torch
from torch.types import Device
from torch.nn.modules.loss import _Loss
import torchtext

from quixer.quixer_model import Quixer
from quixer.baseline_models import Transformer, LSTM, FNet

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer

import random
import os
import time
import math
from tqdm import tqdm
from typing import Any, Optional, Tuple, Callable
from pathlib import Path

import numpy as np

import torch
from torch.types import Device
from torch.nn.modules.loss import _Loss
import torchtext

from quixer.quixer_model import Quixer
from quixer.baseline_models import Transformer, LSTM, FNet

from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer



class Lookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = k

        # Track number of "fast" updates so we know when to do the slow update
        self._step_count = 0

        # Copy of the fast params to “slow” buffer
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = []
            for p in group['params']:
                sp.append(p.clone().detach())
            self.slow_params.append(sp)

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """
        1. Perform one 'fast' step with the base optimizer
        2. Every k steps, update slow weights
        """
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            # Slow update
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    # slow <- slow + alpha * (fast - slow)
                    slow += self.alpha * (p.data - slow)
                    # Then copy back to fast parameters
                    p.data.copy_(slow)

        return loss




class AdaptiveLookahead(torch.optim.Optimizer):
    def __init__(self, base_optimizer,method, alpha=0.5, initial_k=5, k_multiplier=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not initial_k >= 1:
            raise ValueError(f"Invalid initial_k: {initial_k}")

        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.k = initial_k
        self.k_multiplier = k_multiplier
        self._step_count = 0
        self.method = method
        
        # Store initial learning rate to detect changes
        self.last_lr = self.base_optimizer.param_groups[0]['lr']

        # Initialize slow parameter buffers
        self.slow_params = []
        for group in self.base_optimizer.param_groups:
            sp = []
            for p in group['params']:
                sp.append(p.clone().detach())
            self.slow_params.append(sp)

    def check_lr_change(self):
        """Check if learning rate has changed and update k accordingly"""
        current_lr = self.base_optimizer.param_groups[0]['lr']
        if self.method == 'adaptive_decrease':
            if current_lr < self.last_lr:
                self.k = max(1, self.k-self.k_multiplier)
        if self.method == 'adaptive_increase': 
            if current_lr < self.last_lr:
                self.k += self.k_multiplier 
            print(f"\nLearning rate decreased from {self.last_lr:.6f} to {current_lr:.6f}. New k: {self.k}")
        self.last_lr = current_lr

    @property
    def param_groups(self):
        return self.base_optimizer.param_groups

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """
        Performs one step of optimization with adaptive k value:
        1. Check for learning rate changes and update k if needed
        2. Perform one 'fast' step with base optimizer
        3. Every k steps, update slow weights
        """
        # Check for learning rate changes
        self.check_lr_change()
        
        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        # Perform slow weight update if needed
        if self._step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    slow = self.slow_params[group_idx][p_idx]
                    # slow <- slow + alpha * (fast - slow)
                    slow += self.alpha * (p.data - slow)
                    # Copy back to fast parameters
                    p.data.copy_(slow)

        return loss


def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    """
    Computes time elapsed in minutes and seconds when given two UNIX timestamps
    with the starting time and ending time.

    Args:
      start_time: Starting time as a UNIX timestamp.
      end_time: End time as a UNIX timestamp.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
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

    n_batches = iterator.shape[0] - window_size

    idxs = list(range(n_batches))
    random.shuffle(idxs)

    for ctr, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        optimizer.zero_grad()

        yhat, norm_avg = model(x)

        loss = loss_function(yhat, y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / n_batches


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: _Loss,
    window_size: int,
) -> float:
    """
    Evaluates model on the supplied data.
    """

    model.eval()

    epoch_loss = 0

    n_batches = data.shape[0] - window_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            x, y = get_batch_s2s(data, batch_idx, window_size)

            yhat, _ = model(x)

            loss = loss_function(yhat, y)

            epoch_loss += loss.item()

    return epoch_loss / n_batches


def train_cycle(
    model: torch.nn.Module,
    hyperparams: dict[str, Any],
    train_iter: torch.Tensor,
    val_iter: torch.Tensor,
    test_iter: torch.Tensor,
) -> float:
    """
    Run a training cycle.

    Args:
      model: The model to train.
      hyperparams: The model hyperparameters.
      train_iter: Tensor containing training data returned by `setup_dataset` function.
      val_iter: Tensor containing validation data returned by `setup_dataset` function.
      test_iter: Tensor containing test data returned by `setup_dataset` function.
    """

    folder_path = Path("./trained_models")
    folder_path.mkdir(exist_ok=True, parents=True)
    checkpoint_fpath = (
        folder_path
        / f"q_transformer_lm_{hyperparams['model']}_{hyperparams['seed']}_{int(time.time())}.pt"
    )

    # Set up optimizer
    """optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["wd"],
        eps=hyperparams["eps"],
    )"""
   # First, create the base optimizer.
    base_optimizer = torch.optim.Adam(
    model.parameters(),
    lr=hyperparams["lr"],
    weight_decay=hyperparams["wd"],
    eps=hyperparams["eps"],
    )

    # Then, wrap it with AdaptiveLookahead.
    optimizer = AdaptiveLookahead(
    base_optimizer=base_optimizer,
    method='adaptive_increase',  # or 'adaptive_increase' as needed
    alpha=0.5,    # You can adjust these hyperparameters
    initial_k=5,
    k_multiplier=5,
    )

    optimizer = Lookahead(base_optimizer,alpha=0.5,k=10)


    # Set up learning rate scheduler
    scheduler = None
    if hyperparams["lr_sched"] == "cos":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=hyperparams["restart_epochs"]
        )

    loss_function = torch.nn.CrossEntropyLoss()

    def _evaluate(iter: torch.Tensor):
        return evaluate(model, iter, loss_function, hyperparams["window"])

    best_valid_loss = float("inf")
    for epoch in range(hyperparams["epochs"]):
        start_time = time.time()

        train_loss = train_epoch(
            model,
            train_iter,
            optimizer,
            loss_function,
            hyperparams["max_grad_norm"],
            scheduler,
            hyperparams["window"],
        )

        valid_loss = _evaluate(val_iter)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_fpath)

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train ppl: {math.exp(train_loss)}")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")

    model.load_state_dict(torch.load(checkpoint_fpath))

    valid_loss = _evaluate(val_iter)
    test_loss = _evaluate(test_iter)

    print("FINAL TRAINED MODEL STATS:")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. ppl: {math.exp(valid_loss)}")
    print(f"\t Test Loss: {test_loss:.3f} |  Test ppl: {math.exp(test_loss)}")

    return test_loss


def seed(SEED: int) -> None:
    """
    Sets the seed for Python's random module, numpy's RNG and torch's RNG.

    Args:
      SEED: integer specifying the seed
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device) -> Callable:
    """
    Returns a function that runs the training cycle on a specified torch device.

    Args:
      device: Torch device

    Returns:
      Callable taking in a set of parameters as a dict and returning the value of the validation loss
      at the end of the training cycle.
    """

    def train_evaluate(parameterization: dict[str, Any]) -> float:
        """
        Train the model and return the test loss.
        """

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"]
        )

        model = create_model(parameterization, device, len(vocab))

        initialise_weights(model)

        model = model.to(device)

        valid_loss = train_cycle(
            model, parameterization, train_iter, val_iter, test_iter
        )

        return valid_loss

    return train_evaluate

def epoch_time(start_time: float, end_time: float) -> Tuple[float, float]:
    """
    Computes time elapsed in minutes and seconds when given two UNIX timestamps
    with the starting time and ending time.

    Args:
      start_time: Starting time as a UNIX timestamp.
      end_time: End time as a UNIX timestamp.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


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
    raw_dset = load_dataset("ptb_text_only", trust_remote_code=True)

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


def train_epoch(
    model: torch.nn.Module,
    iterator: torch.Tensor,
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

    n_batches = iterator.shape[0] - window_size

    idxs = list(range(n_batches))
    random.shuffle(idxs)

    for ctr, batch_idx in tqdm(enumerate(idxs), total=n_batches):
        x, y = get_batch_s2s(iterator, batch_idx, window_size)
        optimizer.zero_grad()

        yhat, norm_avg = model(x)

        loss = loss_function(yhat, y)
        loss.backward()

        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    return epoch_loss / n_batches


def evaluate(
    model: torch.nn.Module,
    data: torch.Tensor,
    loss_function: _Loss,
    window_size: int,
) -> float:
    """
    Evaluates model on the supplied data.
    """

    model.eval()

    epoch_loss = 0

    n_batches = data.shape[0] - window_size

    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches)):
            x, y = get_batch_s2s(data, batch_idx, window_size)

            yhat, _ = model(x)

            loss = loss_function(yhat, y)

            epoch_loss += loss.item()

    return epoch_loss / n_batches




def seed(SEED: int) -> None:
    """
    Sets the seed for Python's random module, numpy's RNG and torch's RNG.

    Args:
      SEED: integer specifying the seed
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def get_train_evaluate(device: Device) -> Callable:
    """
    Returns a function that runs the training cycle on a specified torch device.

    Args:
      device: Torch device

    Returns:
      Callable taking in a set of parameters as a dict and returning the value of the validation loss
      at the end of the training cycle.
    """

    def train_evaluate(parameterization: dict[str, Any]) -> float:
        """
        Train the model and return the test loss.
        """

        if "seed" not in parameterization:
            parameterization["seed"] = int.from_bytes(os.urandom(4), "big")

        seed(parameterization["seed"])

        vocab, (train_iter, val_iter, test_iter), PAD_TOK = setup_dataset(
            device, parameterization["batch_size"], parameterization["window"]
        )

        model = create_model(parameterization, device, len(vocab))

        initialise_weights(model)

        model = model.to(device)

        valid_loss = train_cycle(
            model, parameterization, train_iter, val_iter, test_iter
        )

        return valid_loss

    return train_evaluate
