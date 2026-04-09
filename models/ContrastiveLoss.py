import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.3, base_weights=[0.8, 1.0, 1.5],
                 cache_size=2000, cache_sample_ratio=0.3,
                 initial_threshold=0.28, final_threshold=0.38,  
                 threshold_ramp_steps=16000,
                 hard_neg_ratio_limit=0.4):  
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.base_weights = torch.tensor(base_weights).cuda()
        self.cache_size = cache_size
        self.cache_sample_ratio = cache_sample_ratio
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.threshold_ramp_steps = threshold_ramp_steps
        self.hard_neg_ratio_limit = hard_neg_ratio_limit 
        
        
        self.hard_neg_threshold = initial_threshold
        self.step_counter = 0
        
        
        self.cache = deque(maxlen=cache_size)
        self.cache_add_count = 0
        self.cache_reject_count = 0

    def forward(self, embeddings, raw_labels):
        
        if self.training:
            self._update_threshold()
        
       
        labels = self._convert_labels(raw_labels)
        valid_mask = labels != -1
        labels = labels[valid_mask]
        embeddings = embeddings[valid_mask]
        
        if embeddings.numel() == 0:
            return torch.tensor(0.0).to(embeddings.device)
        
       
        mined_embeddings, mined_labels, weights = self._fixed_sample_mining(embeddings, labels)
        
       
        loss = self._compute_loss(mined_embeddings, mined_labels, weights)
        
      
        self._log_stats(mined_labels, weights)
        
       
        self.step_counter += 1
        
        return loss

    def _update_threshold(self):
        
        if self.threshold_ramp_steps == 0:
            return
            
       
        progress = min(1.0, self.step_counter / self.threshold_ramp_steps)
        
        sqrt_progress = np.sqrt(progress)
        self.hard_neg_threshold = self.initial_threshold + (
            self.final_threshold - self.initial_threshold) * sqrt_progress

    def _convert_labels(self, raw_labels):
        labels = torch.full_like(raw_labels, -1, dtype=torch.long)
        
        # ID (0 < ID < 5555)
        valid_id_mask = (raw_labels > 0) & (raw_labels < 5555)
        labels[valid_id_mask] = 1
        
        #  (ID=0)
        background_mask = (raw_labels == 0)
        labels[background_mask] = 0
        
        # ID (ID=5555)
        hard_neg_mask = (raw_labels == 5555)
        labels[hard_neg_mask] = 2
        
        return labels

    def _fixed_sample_mining(self, embeddings, labels):
        
        device = embeddings.device
        
       
        pos_mask = labels == 1
        neg_mask = labels == 0
        hard_neg_mask = labels == 2
        
      
        with torch.no_grad():
            pos_embeddings = embeddings[pos_mask].detach()
            neg_embeddings = embeddings[neg_mask].detach()
            hard_neg_embeddings = embeddings[hard_neg_mask].detach()
        
    
        with torch.no_grad():
      
            if pos_embeddings.numel() > 0:
                pos_center = pos_embeddings.mean(dim=0)
            else:
                pos_center = torch.zeros_like(embeddings[0])
            
            
            if neg_embeddings.numel() > 0:
                neg_difficulty = F.cosine_similarity(neg_embeddings, pos_center.unsqueeze(0))
            else:
                neg_difficulty = torch.zeros(0).to(device)
                
           
            if hard_neg_embeddings.numel() > 0:
                hard_neg_difficulty = F.cosine_similarity(hard_neg_embeddings, pos_center.unsqueeze(0))
            else:
                hard_neg_difficulty = torch.zeros(0).to(device)
        
        
        if hard_neg_embeddings.numel() > 0:
            for emb, diff in zip(hard_neg_embeddings, hard_neg_difficulty):
                diff_value = diff.item()
                
              
                base_accept = diff_value > self.hard_neg_threshold
                
               
                diversity_accept = False
                if len(self.cache) < self.cache_size * 0.8:  
                    diversity_accept = random.random() < 0.15
                elif len(self.cache) < self.cache_size * 0.95: 
                    diversity_accept = random.random() < 0.08
                
                if base_accept or diversity_accept:
                    self.cache.append((emb.detach().cpu(), diff_value))
                    self.cache_add_count += 1
                else:
                    self.cache_reject_count += 1
        
       
        sampled_cache_tensor = None
        if len(self.cache) > 0:
           
            total_neg_samples = neg_embeddings.numel() + hard_neg_embeddings.numel()
            max_cache_samples = max(1, int(total_neg_samples * self.cache_sample_ratio))
            max_cache_samples = min(max_cache_samples, len(self.cache), 50) 
            
            if max_cache_samples > 0:
               
                #cache_difficulties = np.array([1.0 / (diff + 0.1) for _, diff in self.cache])
                #sampling_probs = cache_difficulties / cache_difficulties.sum()
                #sampled_indices = np.random.choice(
                 #   len(self.cache), 
                  #  size=max_cache_samples, 
                  #  p=sampling_probs,
                  #  replace=False
               # )
                dists = np.array([1.0 - diff for _, diff in self.cache])      # [0,2]
                cache_difficulties = 1.0 / (dists + 1e-4)                    
                cache_difficulties /= cache_difficulties.sum() 
                sampled_indices = np.random.choice(
                    len(self.cache),
                    size=max_cache_samples,
                    p=cache_difficulties,
                    replace=False
                )
                
                sampled_cache_list = [self.cache[i][0] for i in sampled_indices]
                if sampled_cache_list:
                    sampled_cache_tensor = torch.stack(sampled_cache_list).to(device)
        
       
        all_embeddings = []
        all_labels = []
        
       
        total_target = min(200, embeddings.size(0))  
        max_hard_neg = int(total_target * self.hard_neg_ratio_limit) 
        
        
        if pos_embeddings.numel() > 0:
            all_embeddings.append(pos_embeddings)
            all_labels.append(torch.ones(pos_embeddings.size(0), device=device))
        
       
        if neg_embeddings.numel() > 0:
            neg_target = min(neg_embeddings.size(0), int(total_target * 0.4)) 
            if neg_target < neg_embeddings.size(0):
               
                indices = torch.randperm(neg_embeddings.size(0))[:neg_target]
                sampled_neg = neg_embeddings[indices]
            else:
                sampled_neg = neg_embeddings
            
            all_embeddings.append(sampled_neg)
            all_labels.append(torch.zeros(sampled_neg.size(0), device=device))
        
     
        current_hard_neg = []
        if hard_neg_embeddings.numel() > 0:
            hard_neg_from_batch = min(hard_neg_embeddings.size(0), max_hard_neg // 2)
            if hard_neg_from_batch < hard_neg_embeddings.size(0):
             
                _, indices = torch.topk(hard_neg_difficulty, hard_neg_from_batch, largest=False)
                current_hard_neg.append(hard_neg_embeddings[indices])
            else:
                current_hard_neg.append(hard_neg_embeddings)
        
        if sampled_cache_tensor is not None:
            cache_limit = max_hard_neg - (current_hard_neg[0].size(0) if current_hard_neg else 0)
            if cache_limit > 0 and sampled_cache_tensor.size(0) > cache_limit:
              
                indices = torch.randperm(sampled_cache_tensor.size(0))[:cache_limit]
                sampled_cache_tensor = sampled_cache_tensor[indices]
            
            if cache_limit > 0:
                current_hard_neg.append(sampled_cache_tensor)
        
        if current_hard_neg:
            current_hard_neg = torch.cat(current_hard_neg)
            all_embeddings.append(current_hard_neg)
            all_labels.append(torch.ones(current_hard_neg.size(0), device=device) * 2)
        
     
        if not all_embeddings:
            return embeddings, labels, self.base_weights.clone()
        
        mined_embeddings = torch.cat(all_embeddings)
        mined_labels = torch.cat(all_labels)
        
       
        pos_count = (mined_labels == 1).sum().item()
        neg_count = (mined_labels == 0).sum().item()
        hard_neg_count = (mined_labels == 2).sum().item()
        total = pos_count + neg_count + hard_neg_count
        
        if total == 0:
            weights = self.base_weights.clone()
        else:
            weights = torch.tensor([
                1.0 + (pos_count + hard_neg_count) / total,
                1.0 + (pos_count + neg_count) / total,
                1.0 + (neg_count + hard_neg_count) / total
            ]).to(device)
        
        return mined_embeddings, mined_labels, weights

    def _compute_loss(self, embeddings, labels, weights):
     
        pos_mask = (labels == 1)
        if pos_mask.any():
            pos_embeddings = embeddings[pos_mask]
         
            pos_center = pos_embeddings.mean(dim=0, keepdim=True)
         
            pos_dist = torch.norm(pos_embeddings - pos_center, dim=1)
            pos_loss = weights[2] * torch.mean(pos_dist.pow(2))
        else:
            pos_loss = torch.tensor(0.0).to(embeddings.device)
        

        neg_mask = (labels == 0)
        hard_neg_mask = (labels == 2)
        
    
        all_neg_mask = neg_mask | hard_neg_mask
        neg_embeddings = embeddings[all_neg_mask]
        
        if neg_embeddings.numel() > 0:
           
            with torch.no_grad():
                if pos_mask.any():
                    pos_center = embeddings[pos_mask].mean(dim=0, keepdim=True)
                else:
                    pos_center = torch.zeros(1, embeddings.size(1)).to(embeddings.device)
            
            if torch.isnan(pos_center).any() or torch.isinf(pos_center).any():
                pos_center = torch.zeros(1, embeddings.size(1)).to(embeddings.device)
            
            neg_dist = torch.norm(neg_embeddings - pos_center, dim=1)
            
          
            margin = self.margin
            
            
            if neg_mask[all_neg_mask].any():
                neg_loss = weights[0] * F.relu(margin - neg_dist[neg_mask[all_neg_mask]]).pow(2).mean()
            else:
                neg_loss = torch.tensor(0.0).to(embeddings.device)
                
            if hard_neg_mask[all_neg_mask].any():
                hard_neg_loss = weights[1] * F.relu(margin - neg_dist[hard_neg_mask[all_neg_mask]]).pow(2).mean()
            else:
                hard_neg_loss = torch.tensor(0.0).to(embeddings.device)
        else:
            neg_loss = torch.tensor(0.0).to(embeddings.device)
            hard_neg_loss = torch.tensor(0.0).to(embeddings.device)
        
     
        total_loss = pos_loss + neg_loss + hard_neg_loss
        
     
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            return torch.tensor(0.0).to(embeddings.device)
        
        return total_loss

    def _log_stats(self, labels, weights):
       
        if self.step_counter % 100 == 0:
            pos_count = (labels == 1).sum().item()
            neg_count = (labels == 0).sum().item()
            hard_neg_count = (labels == 2).sum().item()
            total = pos_count + neg_count + hard_neg_count
            


            
           
            cache_avg_diff = np.mean([diff for _, diff in self.cache]) if self.cache else 0

            
          
            self.cache_add_count = 0
            self.cache_reject_count = 0
