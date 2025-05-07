"""
Adapted from nanoGPT by Andrej Karpathy:
https://github.com/karpathy/nanoGPT/blob/master/model.py
Added functions:
- generate_pure_sampling: Generates sequences from the model using pure sampling.
- tokens_to_text: Converts token IDs to text using a vocabulary mapping.
- save_model: Saves the model and configuration to a specified directory.
- from_pretrained: Loads a model from a specified directory with device auto-detection.

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect
import math
import yaml
import os
from pathlib import Path

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
        '''
        This is the standard "generate" function that generate sequences from the model using sampling.
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        
        Args:
            idx: Input tensor of shape (batch_size, sequence_length).
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            eos_token_id: End-of-sequence token ID.
        Returns:
            idx: Generated sequences of shape (batch_size, sequence_length + max_new_tokens).
        '''
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
        """
        Batch generation of sequences using pure sampling without temperature or top-k filtering. Set batch_size = 1 if you want to generate a single sequence.
        Args:
            batch_size: Number of sequences to generate.
            max_length: Maximum length of generated sequences.
            return_logprobs: Whether to return log probabilities of generated tokens.
        Returns:
            sequences: List of generated sequences (token IDs).
            logprobs: List of log probabilities for each generated token (if return_logprobs is True).
        """
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