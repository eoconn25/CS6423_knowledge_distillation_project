import random
import json
import torch
import time

import torch
import random
from sklearn.metrics import f1_score

def evaluate_fitness(config, supernet, val_loader, train_loader, param_constraint, device, teacher_name):
    size_m = get_model_size(teacher_name, config)
    if size_m > param_constraint:
        return 0, size_m

    # recalibrate BN
    supernet.train() 
    with torch.no_grad():
        # run 40 batches of train data to recalibrate
        for i, (images, _) in enumerate(train_loader):
            if i > 40: break
            _ = supernet(images.to(device), config)
    
    # now do validation
    supernet.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i > 70: break 
            logits = supernet(images.to(device), config)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # adds in model size for objective
    efficiency_bonus = 0.05 * (1 - (size_m / param_constraint))
    fitness = f1 + efficiency_bonus

    return fitness, size_m
    

def get_model_size(model_type, config, num_classes=61):
    w = config['width']
    depths = config['depth']
    exps = config.get('exp', [1.0, 1.0, 1.0, 1.0]) # Handle missing exp for non-R50
    total_params = 0
    
    if model_type == 'resnet50':
        base_channels = [64, 128, 256, 512]
        # Standard ResNet50 output expansion is 4 (e.g., 512 -> 2048)
        out_expansion = 4 
        is_bottleneck = True
        deep_stem = False
        max_exp=3.0
    elif model_type == 'resnet10t':
        # "T" variants often start narrower or use different scaling
        # Example: ResNet-10t might use [32, 64, 128, 256] 
        # Check your specific supernet implementation here!
        base_channels = [32, 64, 128, 256] 
        out_expansion = 1
        is_bottleneck = False
        deep_stem = True # ResNet-T uses 3x3 convs in stem
        
    else: # resnet18
        base_channels = [64, 128, 256, 512]
        out_expansion = 1
        is_bottleneck = False
        deep_stem = False

    # --- 2. STEM ---
    stem_width = int(base_channels[0] * w)
    if deep_stem:
        # Three 3x3 convs
        total_params += (3 * stem_width * 3 * 3) + (stem_width * stem_width * 3 * 3) * 2
    else:
        # Standard 7x7
        total_params += (3 * stem_width * 7 * 7)
    
    in_channels = stem_width

    # --- 3. STAGES ---
    for stage_idx, num_blocks in enumerate(depths):
        base_mid = base_channels[stage_idx]
        # Apply width multiplier to the base channels of this stage
        out_channels = int(base_mid * out_expansion * w)
        
        for i in range(num_blocks):
            if is_bottleneck:
                # 1x1 down -> 3x3 mid -> 1x1 up (expansion)
                mid_ch = round(base_mid * w * exps[stage_idx])
                total_params += (in_channels * mid_ch * 1 * 1) 
                total_params += (mid_ch * mid_ch * 3 * 3)
                total_params += (mid_ch * out_channels * 1 * 1)
                
                if i == 0: # Shortcut
                    total_params += (in_channels * out_channels * 1 * 1)
                in_channels = out_channels
            else:
                # Basic Block (3x3 -> 3x3)
                total_params += (in_channels * out_channels * 3 * 3)
                total_params += (out_channels * out_channels * 3 * 3)
                
                if i == 0 and (in_channels != out_channels):
                    total_params += (in_channels * out_channels * 1 * 1)
                in_channels = out_channels

    # --- 4. CLASSIFIER ---
    total_params += (in_channels * num_classes)
    return total_params / 1e6


def run_evolutionary_search(supernet, val_loader, train_loader, device, cfg,
                            generations=20, population_size=40, 
                            param_limit=100.0): # limit in Million params
    
    # initialize population
    #population = []
    population = [get_random_config(cfg) for _ in range(population_size)]
    best_overall = {'f1': -1, 'config': None, 'size': 0}
    
    for gen in range(generations):
        results = []
        print(f"\n--- Generation {gen} ---")
        
        for config in population:
            # evaluate population fitness
            f1, size = evaluate_fitness(config, supernet, val_loader, train_loader, param_limit, device, cfg.teacher_name)
            results.append((f1, size, config))
        
        # sort by F1
        results.sort(key=lambda x: x[0], reverse=True)
        
        # track best version
        if results[0][0] > best_overall['f1']:
            best_overall = {
                'f1': results[0][0],
                'size': results[0][1],
                'config': results[0][2]
            }
            print(f"NEW BEST ARCHITECTURE")
            print(f"F1: {best_overall['f1']:.4f} | Size: {best_overall['size']:.2f}M params")
            print(f"Config: {best_overall['config']}")

        # top 20% become parents
        num_parents = max(2, population_size // 5)
        parents = [r[2] for r in results[:num_parents]]
        
        # crossover & mutation!
        next_gen = parents.copy()
        while len(next_gen) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2, cfg.teacher_name)
            child = mutate(child, 0.2, cfg)
            next_gen.append(child)
            
        population = next_gen

    return best_overall


def get_random_config(config):
    if config.teacher_name == 'resnet50':
        return {
            'res': random.choice(config.supernet_res),
            'width': random.choice(config.supernet_width),
            'depth': [random.choice(config.supernet_depth) for _ in range(4)],
            'exp': [random.choice(config.supernet_expansion) for _ in range(4)]
            }
    
    if config.teacher_name == 'resnet18':
        return {
            'res': random.choice(config.supernet_res),
            'width': random.choice(config.supernet_width),
            'depth': [random.choice(config.supernet_depth) for _ in range(4)],
            }
    
    if config.teacher_name == 'resnet10':
        return {
            'res': random.choice(config.supernet_res),
            'width': random.choice(config.supernet_width),
            'depth': [random.choice(config.supernet_depth) for _ in range(4)],
            }

    

def crossover(config_a, config_b, model_name):
    # mixes successful parents
    new_config = {}
    # randomly pick resolution and width from either parent
    new_config['res'] = random.choice([config_a['res'], config_b['res']])
    new_config['width'] = random.choice([config_a['width'], config_b['width']])
    
    # mix depths and expansions block by block
    new_config['depth'] = [random.choice([d1, d2]) for d1, d2 in zip(config_a['depth'], config_b['depth'])]
    if model_name == 'resnet50':
        new_config['exp'] = [random.choice([e1, e2]) for e1, e2 in zip(config_a['exp'], config_b['exp'])]
    
    return new_config

def mutate(config, mutation_rate=0.2, cfg=None):
   # randomly flips genes to explore
    if random.random() < mutation_rate:
        if cfg.teacher_name == 'resnet50':        
            gene_to_flip = random.choice(['res', 'width', 'depth', 'exp'])
            if gene_to_flip == 'res': config['res'] = random.choice(cfg.supernet_res)
            elif gene_to_flip == 'width': config['width'] = random.choice(cfg.supernet_width)
            elif gene_to_flip == 'depth': config['depth'][random.randint(0,3)] = random.choice(cfg.supernet_depth)
            elif gene_to_flip == 'exp': config['exp'][random.randint(0,3)] = random.choice(cfg.supernet_expansion)
        
        elif cfg.teacher_name == 'resnet18':
            gene_to_flip = random.choice(['res', 'width', 'depth'])
            if gene_to_flip == 'res': config['res'] = random.choice(cfg.supernet_res)
            elif gene_to_flip == 'width': config['width'] = random.choice(cfg.supernet_width)
            elif gene_to_flip == 'depth': config['depth'][random.randint(0,3)] = random.choice(cfg.supernet_depth)
        
        elif cfg.teacher_name == 'resnet10':
            gene_to_flip = random.choice(['res', 'width', 'depth'])
            if gene_to_flip == 'res': config['res'] = random.choice(cfg.supernet_res)
            elif gene_to_flip == 'width': config['width'] = random.choice(cfg.supernet_width)
            elif gene_to_flip == 'depth': config['depth'][random.randint(0,3)] = random.choice(cfg.supernet_depth)

    
    return config



'''
def evaluate_fitness(config, supernet, val_loader, lut, latency_constraint, device):
    """
    Calculates the accuracy and checks latency.
    If latency > constraint, we return a fitness of 0.
    """
    # check expected latency in lookup table
    # we use the average depth/exp to match your simplified lookup table
    avg_d = int(round(sum(config['depth']) / 4))
    avg_e = int(round(sum(config['exp']) / 4))
    key = f"res{config['res']}_w{config['width']}_d{avg_d}_e{avg_e}"
    
    latency = lut.get(key, 999) # Default to huge latency if missing
    
    if latency > latency_constraint:
        return 0, latency # Disqualified!

    # check accuracy over a few batches
    supernet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            if i > 10: break
            images, labels = images.to(device), labels.to(device)
            outputs = supernet(images, config)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy, latency'''


'''
def run_evolutionary_search(supernet, val_loader, lut, device, generations=10, population_size=20, constraint=15.0):
    # initialize population
    population = [get_random_config() for _ in range(population_size)]
    best_overall = None
    
    for gen in range(generations):
        scores = []
        print(f"\n--- Generation {gen} ---")
        
        # evaluate population
        for config in population:
            acc, lat = evaluate_fitness(config, supernet, val_loader, lut, constraint, device)
            scores.append((acc, lat, config))
        
        # sort by accuracy
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # keep track of best model
        if best_overall is None or scores[0][0] > best_overall[0]:
            best_overall = scores[0]
            print(f"NEW BEST: Acc {best_overall[0]:.2f}% | Lat {best_overall[1]:.2f}ms")
            
        # selection, keep top 25% as parents
        parents = [s[2] for s in scores[:population_size // 4]]
        
        # create next generation
        new_population = parents.copy() # Elitism: keep best parents
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population

    return best_overall'''

'''def generate_hardware_lut(supernet, search_space):
    supernet.eval().cuda()
    lut = {}  # dictionary where we'll store observed performance
    
    # iterate through all searchable dimensions
    for res in search_space['res']:
        for width in [0.65, 0.8, 1.0, 1.2]:
            for depth in search_space['depth']:
                for exp in search_space['exp']:
                    config = {
                        'res': res, 'width': width, 
                        'depth': [depth]*4, 'exp': [exp]*4
                    }
                    
                    # Measurement logic
                    dummy_input = torch.randn(1, 1, 224, 224).cuda() # dummy data
                    
                    # warm up gpu
                    for _ in range(10): _ = supernet(dummy_input, config)
                    
                    torch.cuda.synchronize()
                    start = time.time()
                    for _ in range(50): _ = supernet(dummy_input, config)
                    torch.cuda.synchronize()
                    
                    latency = (time.time() - start) / 50 * 1000 # ms
                    vram = torch.cuda.max_memory_allocated() / (1024**2) # MB
                    
                    key = f"r{res}_w{width}_d{depth}_e{exp}"
                    lut[key] = {'latency': latency, 'vram': vram}
                    
    with open("hardware_lut.json", "w") as f:
        json.dump(lut, f)
    return lut


def generate_hardware_lut(supernet, device, save_path="hardware_lut.json"):
    supernet.to(device)
    supernet.eval()  # no more training
    
    # define search space
    resolutions = [128, 160, 190, 224]
    widths = [0.65, 0.8, 1.0, 1.2]
    depths = [2, 3, 4]
    expansions = [3, 4, 6]
    
    lut = {}  # dictionary where we'll save results    
    with torch.no_grad():
        for res in resolutions:
            for w in widths:
                for d in depths:
                    for e in expansions:
                        config = {
                            'res': res,
                            'width': w,
                            'depth': [d, d, d, d], # simplified
                            'exp': [e, e, e, e]
                        }
                        
                        # dummy input with specific resolution
                        dummy_input = torch.randn(1, 3, res, res).to(device)
                        
                        # GPU warmup
                        for _ in range(10):
                            _ = supernet(dummy_input, config)
                        
                        # timing
                        torch.cuda.synchronize()
                        start_time = time.time()
                        
                        iterations = 50
                        for _ in range(iterations):
                            _ = supernet(dummy_input, config)
                            
                        torch.cuda.synchronize()
                        latency = (time.time() - start_time) / iterations * 1000 # ms
                        
                        key = f"res{res}_w{w}_d{d}_e{e}"
                        lut[key] = round(latency, 3)
                        #print(f"Config: {key} | Latency: {latency:.2f}ms")

    with open(save_path, 'w') as f:
        json.dump(lut, f, indent=4)
    
    return lut


'''

'''
import sys
import importlib

# 1. Purge the old version from Python's cache
if 'knowledge_distillation.evolutionary_search' in sys.modules:
    del sys.modules['knowledge_distillation.evolutionary_search']

# 2. Re-import
import knowledge_distillation.evolutionary_search as es

# 3. Double-check if crossover is now visible
if hasattr(es, 'crossover'):
    print("✅ Crossover function found! The ghost has been evicted.")
else:
    print("❌ Still not found. Check if you saved the .py file in the right directory!")'''