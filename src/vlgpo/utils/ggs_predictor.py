from typing import List, Optional, Tuple
from Levenshtein import distance as levenshtein
import numpy as np
import torch
import logging
import os
from omegaconf import DictConfig
import pandas as pd
from models.ggs import BaseCNN
from omegaconf import OmegaConf
import glob
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

from vlgpo.utils.tokenizer import Encoder

to_np = lambda x: x.cpu().detach().numpy()
to_list = lambda x: to_np(x).tolist()

def diversity(seqs):
    num_seqs = len(seqs)
    total_dist = 0
    for i in range(num_seqs):
        for j in range(num_seqs):
            x = seqs[i]
            y = seqs[j]
            if x == y:
                continue
            total_dist += levenshtein(x, y)
    return total_dist / (num_seqs*(num_seqs-1))

class EvalRunner:
    def __init__(self, scenario, runner_cfg):
        self._cfg = runner_cfg
        self._log = logging.getLogger(__name__)
        self.predictor_tokenizer = Encoder()
        gt_csv = pd.read_csv(self._cfg.gt_csv)
        oracle_dir = self._cfg.oracle_dir
        self.use_normalization = self._cfg.use_normalization

        # Read in known sequences and their fitnesses
        self._max_known_score = np.max(gt_csv.score)
        self._min_known_score = np.min(gt_csv.score)
        self.normalize = lambda x: to_np((x - self._min_known_score) / (self._max_known_score - self._min_known_score)).item()
        self._log.info(f'Read in {len(gt_csv)} ground truth sequences.')
        self._log.info(f'Maximum known score {self._max_known_score}.')
        self._log.info(f'Minimum known score {self._min_known_score}.')

        # Read in base pool used to generate sequences.
        base_pool_seqs = pd.read_csv(self._cfg.base_pool_path)
        self._base_pool_seqs = base_pool_seqs.sequence.tolist()
        self.device = 'cpu' 
        oracle_path = os.path.join(oracle_dir, 'oracle_' + str(scenario) + '.ckpt')
        oracle_state_dict = torch.load(oracle_path, map_location=self.device)
        cfg_path = os.path.join(oracle_dir, 'config_' + str(scenario) + '.yaml')
        with open(cfg_path, 'r') as fp:
            ckpt_cfg = OmegaConf.load(fp.name)

        self._cnn_oracle = BaseCNN(**ckpt_cfg.model.predictor) # oracle has same architecture as predictor
        self._cnn_oracle.load_state_dict(
            {k.replace('predictor.', ''): v for k,v in oracle_state_dict['state_dict'].items()})
        self._cnn_oracle = self._cnn_oracle.to(self.device)
        self._cnn_oracle.eval()
        if self._cfg.predictor_dir is not None:
            predictor_path = os.path.join(self._cfg.predictor_dir, 'last.ckpt')
            predictor_state_dict = torch.load(predictor_path, map_location=self.device)
            self._predictor = BaseCNN(**ckpt_cfg.model.predictor) #oracle has same architecture as predictor
            self._predictor.load_state_dict(
                {k.replace('predictor.', ''): v for k,v in predictor_state_dict['state_dict'].items()})
            self._predictor = self._predictor.to(self.device)
        self.run_oracle = self._run_cnn_oracle

    def novelty(self, sampled_seqs):
        # sampled_seqs: top k
        # existing_seqs: range dataset
        all_novelty = []
        for src in sampled_seqs:  
            min_dist = 1e9
            for known in self._base_pool_seqs:
                dist = levenshtein(src, known)
                if dist < min_dist:
                    min_dist = dist
            all_novelty.append(min_dist)
        return all_novelty

    def tokenize(self, seqs):
        return self.predictor_tokenizer.encode(seqs).to(self.device)

    def _run_cnn_oracle(self, seqs):
        tokenized_seqs = self.tokenize(seqs)
        batches = torch.split(tokenized_seqs, self._cfg.batch_size, 0)
        scores = []
        for b in batches:
            if b is None:
                continue
            results = self._cnn_oracle(b).detach()
            scores.append(results)
        return torch.concat(scores, dim=0)
    
    def evaluate_sequences(self, topk_seqs, use_oracle = True):
        topk_seqs = list(set(topk_seqs))
        num_unique_seqs = len(topk_seqs)
        topk_scores = self.run_oracle(topk_seqs) if use_oracle else self.run_predictor(topk_seqs)
        normalized_scores = [self.normalize(x) for x in topk_scores]
        seq_novelty = self.novelty(topk_seqs)
        results_df = pd.DataFrame({
            'sequence': topk_seqs,
            'oracle_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        })  if use_oracle else pd.DataFrame({
            'sequence': topk_seqs,
            'predictor_score': to_list(topk_scores),
            'normalized_score': normalized_scores,
            'novelty': seq_novelty,
        })

        if num_unique_seqs == 1:
            seq_diversity = 0
        else:
            seq_diversity = diversity(topk_seqs)
               
        metrics_scores = normalized_scores if self.use_normalization else topk_scores.detach().cpu().numpy()
        metrics_df = pd.DataFrame({
            'num_unique': [num_unique_seqs],
            'mean_fitness': [np.mean(metrics_scores)],
            'mean_fitness': [np.mean(metrics_scores)],
            'median_fitness': [np.median(metrics_scores)],
            'std_fitness': [np.std(metrics_scores)],
            'max_fitness': [np.max(metrics_scores)],
            'mean_diversity': [seq_diversity],
            'mean_novelty': [np.mean(seq_novelty)],
            'median_novelty': [np.median(seq_novelty)],
        })
        return results_df, metrics_df

def process_baseline_seqs(samples_path, topk):
    """Process baseline samples."""
    df = pd.read_csv(samples_path)
    column_name = 'sequence' if 'sequence' in df.columns else df.columns[0]
    sampled_seqs = df[column_name].tolist()
    return sampled_seqs

def eval_oracle(scenario, task, generated_samples_dir):
    gt_file = scenario + "_ground_truth.csv" 
    base_pool_path = os.path.join(Path(__file__).resolve().parents[3],'data',scenario + "_" + task + ".csv")

    topk = None 
    samples_dir = generated_samples_dir
    _method_fn = lambda x: process_baseline_seqs(x, topk)
    
    cfg = OmegaConf.create({
        "experiment": {
            "gt_csv": os.path.join(Path(__file__).resolve().parents[3],'data',gt_file), 
            "oracle_dir": os.path.join(Path(__file__).resolve().parents[3],'checkpoints', 'oracle'),
            "predictor_dir": None,
        },
        "runner": {
            "batch_size": 128,
            "base_pool_path": base_pool_path,
            "oracle": "cnn",
            "gt_csv": "${experiment.gt_csv}",
            "oracle_dir": "${experiment.oracle_dir}",
            "predictor_dir": "${experiment.predictor_dir}",
            "use_normalization": True
        }
    })

    eval_runner = EvalRunner(scenario, cfg.runner)

    # Glob results to evaluate.
    all_csv_paths = [samples_dir]

    # Run evaluation for each result.
    all_results = []
    all_metrics = []
    all_acceptance_rates = []
    use_oracle = False if cfg.runner.predictor_dir is not None else True
    print('Evaluating generated samples using oracle...')
    for csv_path in all_csv_paths:
        topk_seqs = _method_fn(csv_path)
        csv_results, csv_metrics = eval_runner.evaluate_sequences(topk_seqs, use_oracle=use_oracle)
        csv_results['source_path'] = csv_path
        csv_metrics['source_path'] = csv_path
        all_results.append(csv_results)
        all_metrics.append(csv_metrics)

    all_results = pd.concat(all_results) 
    all_metrics = pd.concat(all_metrics)
    
    return all_results['normalized_score'].median(), all_metrics['mean_diversity'][0].item(), all_metrics['median_novelty'][0].item()