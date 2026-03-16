import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# define our distillation loss - we can latr scale to include feature distillation if needed
'''class DistillationLoss(nn.Module):
    def __init__(self, T, alpha=0.5):
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
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss, soft_loss'''

# distillation based on mse rather than kl divergence
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        # We generally drop T (Temperature) for Logit Matching 
        # because we are comparing raw numbers, not sharpened distributions.
        self.alpha = alpha  

    def forward(self, student_logits, teacher_logits, labels):
        # 1. Soft Loss: Mean Squared Error between raw logits
        # This is the "Tracing" part—matching raw floating point values.
        soft_loss = F.mse_loss(student_logits, teacher_logits)
        
        # 2. Hard Loss: Standard Cross Entropy
        # This ensures the student still learns the ground truth labels.
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # We weight it heavily toward the teacher (alpha=0.9) to pull
        # the student out of that 10% accuracy floor.
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, soft_loss
    
# GreedyNAS path sampler - grabs random parameters for our samples
def sample_configs(num_samples=10):
    configs = []
    for _ in range(num_samples):
        configs.append({
            'res': random.choice([128, 160, 190, 224]),
            'width': random.choice([0.65, 0.8, 1.0, 1.2]),
            'depth': [random.randint(2, 4) for _ in range(4)],
            'exp': [random.choice([1.5, 2, 3]) for _ in range(4)]
        })
    return configs


# function for tracking our training - we want to grab losses, KL divergence, and accuracy over time
class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0
        self.total_kl = 0
        self.correct = 0
        self.samples = 0
        self.steps = 0

    def update(self, loss, kl, logits, labels):
        self.total_loss += loss
        self.total_kl += kl
        _, predicted = torch.max(logits, 1)
        self.correct += (predicted == labels).sum().item()
        self.samples += labels.size(0)
        self.steps += 1

    def get_stats(self):
        return {
            "avg_loss": self.total_loss / self.steps,
            "avg_kl": self.total_kl / self.steps,
            "accuracy": (self.correct / self.samples) * 100
        }

    
# training loop
# will need to pass supernet object, teacher, loader, optimizer, and epoch info
def train_supernet(supernet, teacher, paths_sampled, train_loader, optimizer, epoch, warmup_epochs=10, temperature=4.0, alpha=0.7):
    criterion = DistillationLoss()  #T=temperature, alpha=alpha
    supernet.train()
    teacher.eval() # make sure teacher is frozen
    tracker = MetricTracker()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        
        # run forward pass through teacher to get logits
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        if epoch < warmup_epochs:
            phase = 'warm'
            top_configs = sample_configs(1)  # just sample one path while we warm up - uniform
        else:
            phase = 'greed'
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
            supernet.train()  # set back to train
        
        batch_loss = 0.0
        
        # loop through our best k paths and update their weights
        for config in top_configs:
            student_logits = supernet(images, config)
            # if we add feature distillation, it'll be here
            loss, soft_loss = criterion(student_logits, teacher_logits, labels)
            
            (loss / len(top_configs)).backward()
            batch_loss += loss.item() / len(top_configs)
            
            tracker.update(loss.item(), soft_loss.item(), student_logits, labels)  # update tracker
        
        torch.nn.utils.clip_grad_norm_(supernet.parameters(), max_norm=5.0)
        optimizer.step()

        '''if batch_idx % 10 == 0:
            # now on to Carl, for weather
            print(f"Epoch {epoch} | {phase} | Batch {batch_idx} | Current path loss: {batch_loss:.4f}")'''
        
    return tracker.get_stats(), phase


def validate_supernet(supernet, val_loader, config):
    supernet.eval()
    
    # define min and max configurations for the model
    max_config = {
        'res': 224, 'width': 1.2, 
        'depth': [max(config.supernet_depth)] * 4, 
        'exp': [max(config.supernet_expansion)] * 4
    }
    min_config = {
        'res': 128, 'width': 0.65, 
        'depth': [min(config.supernet_depth)] * 4, 
        'exp': [min(config.supernet_expansion)] * 4
    }
    
    results = {"MAX": {"correct": 0, "loss": 0.0}, "MIN": {"correct": 0, "loss": 0.0}}
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            total_samples += labels.size(0)
            
            for name, sub_config in [("MAX", max_config), ("MIN", min_config)]:
                logits = supernet(images, sub_config)
                loss = F.cross_entropy(logits, labels)
                
                _, predicted = torch.max(logits, 1)
                results[name]["correct"] += (predicted == labels).sum().item()
                results[name]["loss"] += loss.item()

    # report stats
    stats = {}
    for name in ["MAX", "MIN"]:
        stats[name] = {
            "accuracy": (results[name]["correct"] / total_samples) * 100,
            "avg_loss": results[name]["loss"] / len(val_loader)
        }
    
    return stats