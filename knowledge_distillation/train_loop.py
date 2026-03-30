import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# define our distillation loss - we can latr scale to include feature distillation if needed
class DistillationLoss(nn.Module):
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
        hard_loss = F.cross_entropy(student_logits, labels, label_smoothing=0.1)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss, soft_loss


# GreedyNAS path sampler - grabs random parameters for our samples
def sample_configs(config, num_samples=10):
    configs = []
    if config.teacher_name == 'resnet50':
        for _ in range(num_samples):
            configs.append({
                'res': random.choice(config.supernet_res),
                'width': random.choice(config.supernet_width),
                'depth': [random.choice(config.supernet_depth) for _ in range(4)],
                'exp': [random.choice(config.supernet_expansion) for _ in range(4)]
            })
    
    if config.teacher_name == 'resnet18':
        for _ in range(num_samples):
            configs.append({
                'res': random.choice(config.supernet_res),
                'width': random.choice(config.supernet_width),
                'depth': [random.choice(config.supernet_depth) for _ in range(4)],
            })
    
    if config.teacher_name == 'resnet10':
        for _ in range(num_samples):
            configs.append({
                'res': random.choice(config.supernet_res),
                'width': random.choice(config.supernet_width),
                'depth': [random.choice(config.supernet_depth) for _ in range(4)],
            })
            
    return configs


from sklearn.metrics import f1_score
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
        self.all_preds = []
        self.all_labels = []

    def update(self, loss, kl, logits, labels):
        self.total_loss += loss
        self.total_kl += kl
        _, predicted = torch.max(logits, 1)
        
        # is for accuracy
        self.correct += (predicted == labels).sum().item()
        self.samples += labels.size(0)
        self.steps += 1
        
        # is for f1
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

    def get_stats(self):
        macro_f1 = f1_score(self.all_labels, self.all_preds, average='macro', zero_division=0)
        
        return {
            "avg_loss": self.total_loss / self.steps,
            "avg_kl": self.total_kl / self.steps,
            "accuracy": (self.correct / self.samples) * 100,
            "f1_macro": macro_f1
        }

    
# training loop
# will need to pass supernet object, teacher, loader, optimizer, and epoch info
def train_supernet(supernet, teacher, config, train_loader, optimizer, epoch):
    criterion = DistillationLoss(T=config.temperature, alpha=config.alpha)  #T=temperature, alpha=alpha
    supernet.train()
    teacher.eval() # make sure teacher is frozen
    tracker = MetricTracker()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        
        # run forward pass through teacher to get logits
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        if epoch < config.warmup_epochs:
            phase = 'warm'
            top_configs = sample_configs(config, 1)  # just sample one path while we warm up - uniform
        else:
            phase = 'greed'
            # greedily select M paths
            candidate_configs = sample_configs(config, num_samples=config.paths_sampled)  # get configs for paths
            path_scores = []
            
            supernet.eval()  # we'll do a quick forward pass to evaluate their performance
            with torch.no_grad():
                for cfg in candidate_configs:
                    output = supernet(images, cfg)
                    loss = F.cross_entropy(output, labels)
                    path_scores.append((loss.item(), cfg))
            
            # sort paths by lowest loss and grab the top k
            path_scores.sort(key=lambda x: x[0])
            top_configs = [x[1] for x in path_scores[:2]]  # k=2
            supernet.train()  # set back to train
        
        batch_loss = 0.0
        
        # loop through our best k paths and update their weights
        for cfg in top_configs:
            student_logits = supernet(images, cfg)
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



def recalibrate_bn(supernet, train_loader, device, sub_config, num_batches=100):
    supernet.train() # BN updates only happen in train mode
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            # ensure the supernet uses the specific sub-architecture
            supernet(images, sub_config)


    
def validate_supernet(supernet, train_loader, val_loader, config):
    supernet.eval()
    
    if config.teacher_name == 'resnet50':
        # define min and max configurations for the RN50
        max_config = {
            'res': max(config.supernet_res), 
            'width': max(config.supernet_width), 
            'depth': [max(config.supernet_depth)] * 4, 
            'exp': [max(config.supernet_expansion)] * 4
        }
        min_config = {
            'res': min(config.supernet_res), 
            'width': min(config.supernet_width), 
            'depth': [min(config.supernet_depth)] * 4, 
            'exp': [min(config.supernet_expansion)] * 4
        }
    else:
        # we don;t need exp in there for RN18 and RN10t
        max_config = {
            'res': max(config.supernet_res), 
            'width': max(config.supernet_width), 
            'depth': [max(config.supernet_depth)] * 4, 
        }
        min_config = {
            'res': min(config.supernet_res), 
            'width': min(config.supernet_width), 
            'depth': [min(config.supernet_depth)] * 4, 
        }
        
    
    stats = {}

    # We iterate through each path separately so we can recalibrate once per path
    for name, sub_config in [("MAX", max_config), ("MIN", min_config)]:
        
        #recalibrate BN
        print(f"Recalibrating BN for {name} path...")
        recalibrate_bn(supernet, train_loader, config.device, sub_config, num_batches=100)
        
        # validation
        supernet.eval()
        correct = 0
        total_loss = 0.0
        total_samples = 0
        
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                
                # Use the same sub_config we just recalibrated
                logits = supernet(images, sub_config)
                loss = F.cross_entropy(logits, labels)
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                total_samples += labels.size(0)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        # record results for this config
        stats[name] = {
            "accuracy": (correct / total_samples) * 100,
            "avg_loss": total_loss / len(val_loader),
            "f1_macro": f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        }

    return stats


def evaluate_f1(model, cfg, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(config.device), labels.to(config.device)
            # Match resolution
            if images.shape[-1] != cfg['res']:
                images = F.interpolate(images, size=(cfg['res'], cfg['res']), mode='bilinear')
            
            logits = model(images, cfg)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)