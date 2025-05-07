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
import argparse
import os
import yaml

from pcfg_dataset import *
from model import *
from training import *

# ==================== ArgParse ====================

def parse_args_and_config():
    """
    Parse command line arguments and config file, with command line arguments
    taking precedence over config file values.
    
    Returns:
        args: Namespace object containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description="PCFG transformer model training and evaluation")
    
    # Config file argument
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # dataset file
    parser.add_argument("--pcfg_file_name", type=str, default=None,
                        help="the name of the PCFG file to use")
    parser.add_argument("--saved_model_name", type=str, default=None,
                        help="the name of the saved model to use")
    
    # dataset parameters
    parser.add_argument("--train_size", type=int, default=10000,
                        help="Size of the training datastet")
    parser.add_argument("--val_size", type=int, default=2000,
                        help="Size of the validation datastet")
    parser.add_argument("--test_size", type=int, default=2000,
                        help="Size of the test datastet")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for dividing the dataset")
    
    # model parameters
    parser.add_argument("--max_len", type=int, default=50,
                        help="Max length of the sequences to generate")
    parser.add_argument("--n_layer", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_embed", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="General dropout rate")
    
    # training hyperparameters
    parser.add_argument("--max_iter", type=int, default=1200,
                        help="Max number of batches to train on")
    parser.add_argument("--eval_interval", type=int, default=300,
                        help="Number of batch for evaluation result to print out")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="Learning rate during training")
    
    # generation parameters
    parser.add_argument("--num_sen_to_generate", type=int, default=5,
                        help="Numer of sentences to generate with pure sampling")
    parser.add_argument("--generate_max_len", type=int, default=15,
                        help="Max length of the generated sentences")
    parser.add_argument("--return_probs", type=bool, default=True,
                        help="Whether to return logprobs during generation")

    
    # First parse args to check if a config file is provided
    args, unknown = parser.parse_known_args()
    
    # Load config from file if provided
    config_dict = {}
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    
    # Update default values with config file values
    parser.set_defaults(**config_dict)
    
    # Parse again with updated defaults
    args = parser.parse_args()
    
    return args

# ==================== Main Execution ====================

def main():
    ###################### Load PCFG and Create Dataset ######################
    args = parse_args_and_config()
    pcfg_file = f"pcfg/{args.pcfg_file_name}"  # Replace with the current PCFG file
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
    train_dataset = PCFGDataset(pcfg, num_samples=args.train_size, max_length=50, 
                              token_to_idx=token_to_idx, special_tokens=special_tokens)
    val_dataset = PCFGDataset(pcfg, num_samples=args.val_size, max_length=50,
                            token_to_idx=token_to_idx, special_tokens=special_tokens)
    test_dataset = PCFGDataset(pcfg, num_samples=args.test_size, max_length=50,
                             token_to_idx=token_to_idx, special_tokens=special_tokens)
    
    # Create data loaders
    batch_size = args.batch_size
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
        block_size=args.max_len,  # Should match max_length in dataset
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embed,
        dropout=args.dropout,
        bias=True,
        special_tokens=SpecialTokens()
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(model_config).to(device)
    
    # Training config
    train_config = {
        'max_iters': args.max_iter,
        'eval_interval': args.eval_interval,
        'learning_rate': args.lr,
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
    model.save_model(f"my_models/{args.saved_model_name}")
    
    ###################### Load pretrained model ######################
    loaded_model = Transformer.from_pretrained(f"my_models/{args.saved_model_name}", device='cpu')
    
    ###################### Example generation with pure sampling and logprobs ######################
    sequences, logprobs = loaded_model.generate_pure_sampling(
        batch_size=args.num_sen_to_generate, 
        max_length=args.generate_max_len,
        return_logprobs=args.return_probs
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