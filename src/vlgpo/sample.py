import os 
import numpy as np
from random import choices
import random
from tqdm import tqdm
import pandas as pd
import re 
from collections import Counter
import math 
import argparse

from torch import nn, optim
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from vlgpo.utils.tokenizer import Encoder
from vlgpo.utils.config_loader import load_config
from vlgpo.utils.ggs_predictor import eval_oracle
from vlgpo.models.ggs import BaseCNN
from vlgpo.models import vae, unet_1d

def check_duplicates(sequences):
    """
    Check if there are duplicates in the list of sequences.
    """
    sequence_counts = Counter(sequences)
    return list(sequence_counts.values()), list(sequence_counts.keys())

def decode(x,tokenizer):
    if x.shape[-1] == 240:
        x = x[...,:-3]
    return tokenizer.decode(x) 

def seed_all(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

def posterior_sample(n_samples, top_k, latent_dim, ode_steps, alpha, J, f_min, f_max, flow_model, vae_model, predictor_model, tokenizer, device, seed):
    seed_all(seed) 

    z0 = torch.randn(n_samples,1,latent_dim).float().to(device)
    cond_fitness = torch.ones((n_samples,1)).float().to(device) 

    t = 0.0
    t_all = np.linspace(0,1,ode_steps+1)
    h = 1./(ode_steps)
    z = z0.clone()

    with torch.no_grad():
        for t_idx, t in enumerate(t_all[:-1]):
            h = (t_all[t_idx + 1] - t_all[t_idx]).item()
            t_curr = torch.ones(z.shape[0], device=z.device) * t

            # Prior step
            v = flow_model(z,t_curr)

            z = z + h*v 

            # Likelihood step
            for j in range(J):
                with torch.enable_grad():
                    z.requires_grad_(True)
                    
                    # Manifold constraint
                    v = flow_model(z,t_curr)
                    ipt_pred = vae_model.decode(z + (1-t-h)*v)

                    if ipt_pred.shape[1] > 28: 
                        ipt_pred = ipt_pred[:,:-3,:] 
                    f_pred = (predictor_model.forward_soft(ipt_pred)- f_min)/(f_max-f_min) 
                    loss_data = 0.5*((f_pred - cond_fitness.squeeze())**2)
                    grad_z = torch.autograd.grad(loss_data,z,torch.ones_like(f_pred))[0]

                z.detach_()
                del v, ipt_pred, f_pred, loss_data
                torch.cuda.empty_cache() 
                
                z = z - h*alpha*grad_z

    # Convert back to sequences
    with torch.no_grad():
        logits = vae_model.decode(z.squeeze())
    preds = logits.argmax(-1)
    generated_sequences = decode(preds,tokenizer)
    
    sequence_values, sequence_instances = check_duplicates(generated_sequences)
    generated_sequences = sequence_instances

    fitness_pred = []
    with torch.no_grad():
        for seq in generated_sequences:
            f = predictor_model.forward(tokenizer.encode(seq).to(device)[None,...]).item()
            fitness_pred.append(f)
    
    fitness_pred = torch.from_numpy(np.asarray(fitness_pred))
    top_k = np.minimum(top_k,len(generated_sequences))
    _, top_k_indices = torch.topk(fitness_pred.flatten(), top_k, dim=0)

    top_k_indices_list = top_k_indices.tolist()
    reordered_sequences = [generated_sequences[i] for i in top_k_indices_list]
    return reordered_sequences

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'
    parser = argparse.ArgumentParser(description="Run sampling with a specific configuration.")
    parser.add_argument(
        "setting",
        nargs="?",                           
        default="aav_medium",               
        choices=["aav_medium", "aav_hard", "gfp_medium", "gfp_hard"],
        help="Choose the sampling setting."
    )
    args = parser.parse_args()

    # Load the corresponding configuration
    config = load_config(args.setting)

    # Set min/max fitness values to normalize predictor outputs
    if "aav" in args.setting:
        f_min, f_max = 0.0, 19.5365
    elif "gfp"  in args.setting:
        f_min, f_max = 1.2834, 4.1231

    # Load tokenizer 
    tokenizer = Encoder()

    # Load VAE encoder/decoder
    encoder_path = Path(__file__).resolve().parents[2] / 'checkpoints/vae/' / f"vae_{args.setting}.pt"
    vae_model = vae.VAE(input_dim=config['prior']['seq_len'],latent_dim=config['prior']['latent_dim']).to(device)
    vae_model.load_state_dict(torch.load(encoder_path, map_location=device)["state_dict"], strict=True)
    vae_model.eval()

    # Load Flow Matching prior 
    prior_path = Path(__file__).resolve().parents[2] / 'checkpoints/flow-prior/' / f"flow_{args.setting}.pt"
    flow_model = unet_1d.Unet1D(dim=config['prior']['dim'],dim_mults=(1,2),channels=1).to(device)
    flow_model.load_state_dict(torch.load(prior_path, map_location=device)["state_dict"], strict=True)
    flow_model.eval()

    # Load predictor
    predictor_path = Path(__file__).resolve().parents[2] / 'checkpoints/predictor/' f"{config['predictor']['type']}" / f"predictor_{args.setting}.ckpt"
    predictor_model = BaseCNN().to(device)
    predictor_model.load_state_dict({k.replace('predictor.', ''): v for k,v in torch.load(predictor_path, map_location=device)['state_dict'].items()})
    predictor_model.eval()

    # VLGPO sampling
    np.random.seed(41)
    seeds = np.random.randint(0,1000,size=5) 
    fitness_all, diversity_all, novelty_all = np.zeros((len(seeds))),  np.zeros((len(seeds))),  np.zeros((len(seeds)))

    for seed_idx, seed in enumerate(seeds):
        seed = seed.item()

        generated_sequences = posterior_sample(
            n_samples=config['sampling']['n_samples'], 
            top_k=config['sampling']['top_k'], 
            latent_dim=config['prior']['latent_dim'], 
            ode_steps=config['sampling']['ode_steps'], 
            alpha=config['sampling']['alpha'][config['predictor']['type']], 
            J=config['sampling']['J'][config['predictor']['type']], 
            f_min=f_min, 
            f_max=f_max, 
            flow_model=flow_model,
            vae_model=vae_model, 
            predictor_model=predictor_model, 
            tokenizer=tokenizer,
            device=device, 
            seed=seed)

        # Save sequences and evaluate using oracle
        output_directory = os.path.join(Path(__file__).resolve().parents[0],'out')
        os.makedirs(output_directory, exist_ok=True)
        file_name = args.setting + ".csv"
        df = pd.DataFrame(generated_sequences, columns=["sequence"]) 
        output_path = os.path.join(output_directory, file_name)
        df.to_csv(output_path, index=False)

        fitness, diversity, novelty = eval_oracle(
            scenario=args.setting.split('_')[0], 
            task=args.setting.split('_')[1],   
            generated_samples_dir=output_path)

        fitness_all[seed_idx] = fitness 
        diversity_all[seed_idx] = diversity
        novelty_all[seed_idx] = novelty

    print('\nVLGPO sampling for %s %s with predictor %s:' %(args.setting.split('_')[0], args.setting.split('_')[1], config['predictor']['type']))
    print('Fitness %.2f (%.2f)' %(fitness_all.mean(),fitness_all.std()))
    print('Diversity %.2f (%.2f)' %(diversity_all.mean(),diversity_all.std()))
    print('Novelty %.2f (%.2f)' %(novelty_all.mean(),novelty_all.std()))
