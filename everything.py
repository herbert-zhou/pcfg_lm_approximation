from dataclasses import dataclass, field
from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
from typing import List, Dict, Tuple
import math
import torch.nn as nn
from torch.nn import functional as F
import inspect
import yaml
import os
from pathlib import Path

# ==================== PCFG Parser and Dataset Generator ====================
class PCFG:
    def __init__(self):
        self.rules = defaultdict(list)
        self.non_terminals = set()
        self.terminals = set()
        self.start_symbol = None
        
    @classmethod
    def from_file(cls, file_path: str):
        pcfg = cls()
        all_symbols = set()
        
        # First pass: collect all non-terminals (LHS of rules)
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                prob = float(parts[0])
                lhs = parts[1] # LHS is the second part of the rule
                rhs = tuple(parts[2:])
                
                pcfg.non_terminals.add(lhs)
                all_symbols.add(lhs)
                all_symbols.update(list(rhs))  # Add all RHS symbols
                
                pcfg.rules[lhs].append((prob, rhs))
                
        pcfg.terminals = all_symbols - pcfg.non_terminals
        
        # Determine start symbol (first non-terminal in first rule)
        if pcfg.rules:
            pcfg.start_symbol = next(iter(pcfg.rules.keys()))
        
        return pcfg
    
    def generate(self, symbol: str = None, max_depth: int = 10) -> List[str]:
        if symbol is None:
            symbol = self.start_symbol
        if max_depth <= 0:
            return []
            
        if symbol in self.terminals:
            return [symbol]
            
        # Select a rule randomly according to probabilities
        possible_rules = self.rules.get(symbol, [])
        if not possible_rules:
            return []
            
        probs, rhs_list = zip(*possible_rules)
        probs = np.array(probs)
        probs /= probs.sum()  # (redundant step for a well-defined PCFG) Normalize probabilities
        
        selected_idx = np.random.choice(len(possible_rules), p=probs)
        selected_rhs = rhs_list[selected_idx]
        
        result = []
        for s in selected_rhs:
            result.extend(self.generate(s, max_depth-1))
        return result

class PCFGDataset(Dataset):
    def __init__(self, pcfg: PCFG, num_samples: int, max_length: int, 
                 token_to_idx: Dict[str, int], special_tokens: Dict[str, int]):
        self.pcfg = pcfg
        self.max_length = max_length
        self.token_to_idx = token_to_idx
        self.special_tokens = special_tokens
        self.samples = []
        
        # Generate samples
        for _ in range(num_samples):
            while True:
                sequence = self.pcfg.generate(max_depth=20)
                if 0 < len(sequence) <= max_length:
                    break
            self.samples.append(sequence)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        # Convert to token indices
        token_ids = [self.token_to_idx[token] for token in sequence]
        
        # Add BOS and EOS tokens
        token_ids = [self.special_tokens['bos']] + token_ids + [self.special_tokens['eos']]
        
        # Create input and target (shifted by one)
        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)
        
        return x, y

def create_token_mappings(pcfg: PCFG, special_tokens: Dict[str, int]) -> Dict[str, int]:
    """Create mappings between tokens and indices."""
    # All terminals become tokens
    tokens = sorted(pcfg.terminals)
    
    # Create vocabulary
    token_to_idx = {token: idx + len(special_tokens) for idx, token in enumerate(tokens)}
    
    # Add special tokens
    for token, idx in special_tokens.items():
        token_to_idx[token] = idx
    
    idx_to_token = {idx: token for token, idx in token_to_idx.items()}
    
    return token_to_idx, idx_to_token

def pcfg_collate_fn(batch, pad_idx: int):
    """Collate function for DataLoader that handles padding."""
    xs, ys = zip(*batch)
    
    # Find max length in this batch
    max_len = max(x.size(0) for x in xs)
    
    # Pad sequences
    padded_xs = []
    padded_ys = []
    masks = []
    
    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)
        
        # Pad input and target
        padded_x = torch.cat([x, torch.full((pad_len,), pad_idx, dtype=torch.long)])
        padded_y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])  # -1 will be ignored in loss
        
        # Create padding mask (1 for real tokens, 0 for padding)
        mask = torch.cat([
            torch.ones(x.size(0), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        padded_xs.append(padded_x)
        padded_ys.append(padded_y)
        masks.append(mask)
    
    return torch.stack(padded_xs), torch.stack(padded_ys), torch.stack(masks)

# ==================== Decoder-only Transformer Implementation (from NanoGPT) ====================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Register causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x, padding_mask=None):
        B, T, C = x.size()
        
        # Project to query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            att = att.masked_fill(padding_mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        # Combine heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, padding_mask=None):
        x = x + self.attn(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Weight tying AFTER initialization
        self.lm_head.weight = self.transformer.wte.weight  # Tie weights
        
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None, padding_mask=None):
        device = idx.device
        b, t = idx.size()
        
        # Position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, padding_mask)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if eos_token_id is not None and (idx_next == eos_token_id).any():
                break
                
        return idx
    
    @torch.no_grad()
    def generate_pure_sampling(self, batch_size: int = 1, max_length: int = 50, return_logprobs: bool = False):
        """Generate sequences from BOS token with proper batch handling and logprobs"""
        device = next(self.parameters()).device
        bos_token = self.config.special_tokens.bos
        eos_token = self.config.special_tokens.eos
        
        # Initialize with BOS token for all sequences
        generated = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # For storing logprobs
        all_logprobs = torch.zeros(batch_size, max_length, device=device) if return_logprobs else None
        active_indices = torch.arange(batch_size, device=device)  # Track original positions
        
        for step in range(max_length):
            if not active.any():
                break
                
            # Get logits for active sequences
            logits, _ = self(generated[active])
            last_logits = logits[:, -1, :]  # (active_count, vocab_size)
            
            # Convert to probabilities and sample
            probs = F.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (active_count, 1)
            
            # Store logprobs if requested
            if return_logprobs:
                log_probs = F.log_softmax(last_logits, dim=-1)
                selected_log_probs = log_probs.gather(1, next_tokens)  # (active_count, 1)
                all_logprobs[active_indices[active], step] = selected_log_probs.squeeze(-1)
            
            # Create update tensor for all sequences
            update = torch.full((batch_size, 1), eos_token, dtype=torch.long, device=device)
            update[active] = next_tokens
            
            # Append to generated sequences
            generated = torch.cat([generated, update], dim=1)
            
            # Update active status
            active &= (update.squeeze(1) != eos_token)
        
        # Convert to list of sequences (remove padding)
        sequences = []
        seq_lengths = []
        for seq in generated:
            eos_pos = (seq == eos_token).nonzero()
            end_pos = eos_pos[0].item() if eos_pos.numel() > 0 else len(seq)
            sequences.append(seq[:end_pos].tolist())
            seq_lengths.append(end_pos)
        
        if return_logprobs:
            # Trim logprobs to actual sequence lengths
            trimmed_logprobs = []
            for i, length in enumerate(seq_lengths):
                if length > 1:  # At least one token after BOS
                    trimmed_logprobs.append(all_logprobs[i, :length-1])  # Exclude BOS
                else:
                    trimmed_logprobs.append(torch.tensor([], device=device))
            return sequences, trimmed_logprobs
        
        return sequences

    def tokens_to_text(self, token_ids_list, idx_to_token):
        """Convert list of token IDs to text using vocabulary mapping"""
        return [
            [idx_to_token.get(idx, "<unk>") for idx in token_ids] 
            for token_ids in token_ids_list
        ]

    def save_model(self, save_dir: str):
        """Save model and config to directory"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # Save config as YAML
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'block_size': self.config.block_size,
            'n_layer': self.config.n_layer,
            'n_head': self.config.n_head,
            'n_embd': self.config.n_embd,
            'dropout': self.config.dropout,
            'bias': self.config.bias,
            'special_tokens': {
                'pad': self.config.special_tokens.pad,
                'bos': self.config.special_tokens.bos,
                'eos': self.config.special_tokens.eos
            }
        }
        
        with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
            yaml.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, save_dir: str, device=None):
        """Load model from directory with device auto-detection"""
        # Load config
        with open(os.path.join(save_dir, "config.yaml")) as f:
            config_dict = yaml.safe_load(f)
        
        # Auto-select device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Recreate config
        special_tokens = SpecialTokens(
            pad=config_dict['special_tokens']['pad'],
            bos=config_dict['special_tokens']['bos'],
            eos=config_dict['special_tokens']['eos']
        )
        
        config = ModelConfig(
            vocab_size=config_dict['vocab_size'],
            block_size=config_dict['block_size'],
            n_layer=config_dict['n_layer'],
            n_head=config_dict['n_head'],
            n_embd=config_dict['n_embd'],
            dropout=config_dict['dropout'],
            bias=config_dict['bias'],
            special_tokens=special_tokens
        )
        
        # Create model on CPU first
        model = cls(config).to('cpu')
        
        # Load weights with proper device mapping
        state_dict = torch.load(
            os.path.join(save_dir, "model.pt"),
            map_location=torch.device('cpu')
        )
        
        # Handle potential weight tying
        if 'lm_head.weight' not in state_dict:
            state_dict['lm_head.weight'] = state_dict['wte.weight']
        
        model.load_state_dict(state_dict)
        
        # Move to final device
        model = model.to(device)
        
        return model

# ==================== Model Configuration ====================

@dataclass
class SpecialTokens:
    pad: int = 0
    bos: int = 1
    eos: int = 2

@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    # special_tokens: SpecialTokens = SpecialTokens()
    special_tokens: SpecialTokens = field(default_factory=SpecialTokens)

# ==================== Training Loop ====================

def train_model(model, train_loader, val_loader, config):
    # Configure optimizer (with nanoGPT's version)
    optimizer = model.configure_optimizers(
        weight_decay=config['weight_decay'],
        learning_rate=config['learning_rate'],
        betas=config['betas'],
        device_type='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Create iterator outside the loop
    train_iter = iter(train_loader)
    
    for step in range(config['max_iters']):
        # Evaluation
        if step % config['eval_interval'] == 0 or step == config['max_iters'] - 1:
            model.eval()
            with torch.no_grad():
                train_loss = evaluate_model(model, train_loader, config['device'])
                val_loss = evaluate_model(model, val_loader, config['device'])
                
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optional: Save model checkpoint
                # torch.save(model.state_dict(), 'best_model.pth')
            
            model.train()
        
        # Training step
        try:
            xb, yb, mask = next(train_iter)
        except StopIteration:
            # Reset iterator if we've exhausted the dataset
            train_iter = iter(train_loader)
            xb, yb, mask = next(train_iter)
        
        xb, yb, mask = xb.to(config['device']), yb.to(config['device']), mask.to(config['device'])
        
        # Forward pass
        _, loss = model(xb, targets=yb, padding_mask=mask)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if config['grad_clip'] != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
    
    return train_losses, val_losses
        

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    
    with torch.no_grad():
        for xb, yb, mask in data_loader:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            _, loss = model(xb, targets=yb, padding_mask=mask)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
    
    model.train()
    return total_loss / total_items if total_items > 0 else float('inf')

# ==================== Main Execution ====================

def main():
    ###################### Load PCFG and Create Dataset ######################
    pcfg_file = "pcfg_bigger.txt"  # Replace with the current PCFG file
    special_tokens = {
        'pad': 0,
        'bos': 1,
        'eos': 2
    }

    # Load PCFG
    pcfg = PCFG.from_file(pcfg_file)

    # Create token mappings
    token_to_idx, idx_to_token = create_token_mappings(pcfg, special_tokens)
    vocab_size = len(token_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Terminals: {len(pcfg.terminals)}")
    print(f"Non-terminals: {len(pcfg.non_terminals)}")
    
    # Create datasets
    train_dataset = PCFGDataset(pcfg, num_samples=10000, max_length=50, 
                              token_to_idx=token_to_idx, special_tokens=special_tokens)
    val_dataset = PCFGDataset(pcfg, num_samples=2000, max_length=50,
                            token_to_idx=token_to_idx, special_tokens=special_tokens)
    test_dataset = PCFGDataset(pcfg, num_samples=2000, max_length=50,
                             token_to_idx=token_to_idx, special_tokens=special_tokens)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=lambda b: pcfg_collate_fn(b, pad_idx=special_tokens['pad']))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=lambda b: pcfg_collate_fn(b, pad_idx=special_tokens['pad']))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda b: pcfg_collate_fn(b, pad_idx=special_tokens['pad']))
    print(f'One epoch of training is {len(train_loader)} iterations/batches.')
    
    ###################### Define model and training configurations ######################
    # Model config
    model_config = ModelConfig(
        vocab_size=vocab_size,
        block_size=50,  # Should match max_length in dataset
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        bias=True,
        special_tokens=SpecialTokens()
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(model_config).to(device)
    
    # Training config
    train_config = {
        'max_iters': 1200,
        'eval_interval': 300,
        'learning_rate': 6e-4,
        'weight_decay': 0.1,
        'betas': (0.9, 0.95),
        'device': device,
        'grad_clip': 1.0
    }
    
    ###################### Training and Evaluations ######################
    # Train the model
    train_model(model, train_loader, val_loader, train_config)
    
    # Test the model
    test_loss = evaluate_model(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")
    
    # Saving the model
    model.save_model("my_models/model_1")
    
    ###################### Example generation with pure sampling and logprobs ######################
    loaded_model = Transformer.from_pretrained("my_models/model_1", device='cpu')
    
    sequences, logprobs = loaded_model.generate_pure_sampling(
        batch_size=5, 
        max_length=15,
        return_logprobs=True
    )
    texts = loaded_model.tokens_to_text(sequences, idx_to_token)
    for text, probs in zip(texts, logprobs):
        print("Sentence:", " ".join(text))
        print("Logprobs:", probs)
    
    ###################### Example generation ######################
    # model.eval()
    # with torch.no_grad():
    #     # Get a sample from test set
    #     x, y, _ = next(iter(test_loader))
    #     x = x[:1].to(device)  # Take first sample from batch
        
    #     # Generate continuation
    #     generated = model.generate(
    #         x,
    #         max_new_tokens=20,
    #         temperature=0.8,
    #         eos_token_id=special_tokens['eos']
    #     )
        
    #     # Convert to tokens
    #     input_tokens = [idx_to_token[idx.item()] for idx in x[0] if idx.item() in idx_to_token]
    #     generated_tokens = [idx_to_token[idx.item()] for idx in generated[0] if idx.item() in idx_to_token]
        
    #     print("\nExample generation:")
    #     print("Input:", " ".join(input_tokens))
    #     print("Generated:", " ".join(generated_tokens))

if __name__ == "__main__":
    main()