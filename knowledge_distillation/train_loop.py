import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# define our distillation loss - we can latr scale to include feature distillation if needed
class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.5):
        super().__init__()
        self.T = T  # temp param for scaling our ditribution
        self.alpha = alpha  # how we'll weight our soft loss

    def forward(self, student_logits, teacher_logits, labels):
        # soft loss - distribution scaled by our temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean'
        ) * (self.T ** 2)
        
        # hard loss - just cross entropy!
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# GreedyNAS path sampler - grabs random parameters for our samples
def sample_configs(num_samples=10):
    configs = []
    for _ in range(num_samples):
        configs.append({
            'res': random.choice([128, 160, 190, 224]),
            'width': random.choice([0.65, 0.8, 1.0, 1.2]),
            'depth': [random.randint(2, 4) for _ in range(4)],
            'exp': [random.choice([3, 4, 6]) for _ in range(4)]
        })
    return configs

# training loop
# will need to pass supernet object, teacher, loader, optimizer, and epoch info
def train_supernet(supernet, teacher, paths_sampled, train_loader, optimizer, epoch, warmup_epochs=10):
    criterion = DistillationLoss(T=4.0, alpha=0.7)
    supernet.train()
    teacher.eval() # make sure teacher is frozen

    for batch_idx, (images, labels) in enumerate(train_loader):
        best_path_loss = 0.0
        images, labels = images.cuda(), labels.cuda()
        
        # run forward pass through teacher to get logits
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        if epoch < warmup_epochs:
            top_configs = sample_configs(1)  # just sample one path while we warm up
            
            optimizer.zero_grad()
            student_logits = supernet(images, top_configs[0])
            loss = criterion(student_logits, teacher_logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=5.0)
            optimizer.step()
            
            best_path_loss = loss.item()
        else:
            # greedily select M paths
            candidate_configs = sample_configs(num_samples=paths_sampled)  # get configs for paths
            path_scores = []
            
            supernet.eval()  # we'll do a quick forward pass to evaluate their performance
            with torch.no_grad():
                for config in candidate_configs:
                    output = supernet(images, config)
                    loss = F.cross_entropy(output, labels)
                    path_scores.append((loss.item(), config))
            
            # sort paths by lowest loss and grab the top k
            path_scores.sort(key=lambda x: x[0])
            top_configs = [x[1] for x in path_scores[:2]]  # k=2
            best_path_loss = path_scores[0][0]
        
        # set back to train, zero grads
        supernet.train()
        optimizer.zero_grad()
        
        # loop through our best k paths and update their weights
        for config in top_configs:
            student_logits = supernet(images, config)
            # if we add feature distillation, it'll be here
            loss = criterion(student_logits, teacher_logits, labels)
            loss.backward()  # accumulate grads for all k paths
        
        torch.nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=5.0)
        optimizer.step()

        if batch_idx % 10 == 0:
            # now on to Carl, for weather
            phase = "warm" if epoch < warmup_epochs else "greed"
            print(f"Epoch {epoch} | {phase} | Batch {batch_idx}: current path loss is {best_path_loss:.4f}")
        
        return best_path_loss