import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================== RESNET 10 EXTRACTION FUNCTIONS==================================
class StandaloneResNet10(nn.Module):
    def __init__(self, config, num_classes=61):
        super().__init__()
        self.res = config['res']
        w = config['width']
        depths = config['depth']
        
        # --- RESNET-T DEEP STEM ---
        # Matches your supernet: 3x 3x3 convs
        self.stem = nn.Sequential(
            nn.Conv2d(3, int(32*w), kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(int(32*w)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32*w), int(32*w), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(32*w)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(32*w), int(64*w), kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(int(64*w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # --- STAGES ---
        self.stages = nn.ModuleList()
        in_c = int(64 * w)
        base_channels = [64, 128, 256, 512]
        
        for i, d in enumerate(depths):
            out_c = int(base_channels[i] * w)
            blocks = []
            for j in range(d):
                # Stage 0 is stride 1, others are stride 2 (per your supernet logic)
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(StaticBasicBlock(in_c, out_c, stride))
                in_c = out_c
            self.stages.append(nn.Sequential(*blocks))
            
        self.classifier = nn.Linear(in_c, num_classes)

    def forward(self, x):
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode='bilinear', align_corners=False)
        
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)
    
    
def extract_resnet10(supernet, config):
    # 1. Initialize the static structure
    student = StandaloneResNet10(config)
    student.eval()
    
    super_state = supernet.state_dict()
    student_state = student.state_dict()
    
    print(f"Slicing weights for width {config['width']} and depth {config['depth']}...")
    
    # 2. Iterate through student parameters and pull from supernet
    for name, param in student_state.items():
        if name in super_state:
            src = super_state[name]
            s = param.shape
            
            # Perform the slice
            if len(s) == 4: # Conv
                param.copy_(src[:s[0], :s[1], :s[2], :s[3]])
            elif len(s) == 2: # Linear
                param.copy_(src[:s[0], :s[1]])
            elif len(s) == 1: # Bias/BN
                param.copy_(src[:s[0]])
        else:
            print(f"⚠️ Warning: {name} not found in supernet. Check architecture matching.")

    student.load_state_dict(student_state)
    return student


# ============================== resnet 18 functions =======================
class StaticBasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # Match the supernet: CustomConv defaults to bias=True
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class StandaloneResNet18(nn.Module):
    def __init__(self, config, num_classes=61):
        super().__init__()
        self.res = config['res']
        w = config['width']
        depths = config['depth']
        
        # Stem
        curr_c = int(64 * w)
        self.stem = nn.Sequential(
            nn.Conv2d(3, curr_c, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(curr_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stages
        self.stages = nn.ModuleList()
        base_channels = [64, 128, 256, 512]
        
        for i, d in enumerate(depths):
            out_c = int(base_channels[i] * w)
            blocks = []
            for j in range(d):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(StaticBasicBlock(curr_c, out_c, stride))
                curr_c = out_c
            self.stages.append(nn.Sequential(*blocks))
            
        self.classifier = nn.Linear(curr_c, num_classes)

    def forward(self, x):
        # Handle resolution internally if needed, or rely on external transforms
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode='bilinear', align_corners=False)
        
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)

def extract_resnet18(supernet, config):
    # 1. Instantiate the static student
    student = StandaloneResNet18(config)
    student.eval()
    
    super_state = supernet.state_dict()
    student_state = student.state_dict()
    
    print("Transitioning weights...")
    
    for name, param in student_state.items():
        # The naming convention for stages in Supernet is 'stages.i.j.layer'
        # In Static (Sequential), it is also 'stages.i.j.layer'
        # This allows direct name matching for the blocks that exist.
        if name in super_state:
            source_weight = super_state[name]
            s = param.shape # Student shape (the slice)
            
            # Slicing logic based on dimensionality
            if len(s) == 4: # Conv layers: [out, in, h, w]
                param.copy_(source_weight[:s[0], :s[1], :s[2], :s[3]])
            elif len(s) == 2: # Linear layers: [out, in]
                param.copy_(source_weight[:s[0], :s[1]])
            elif len(s) == 1: # Bias or BN layers: [out]
                param.copy_(source_weight[:s[0]])
    
    student.load_state_dict(student_state)
    return student



# ======================= resnet50 extraction tools ====================
class StaticBottleneck(nn.Module):
    def __init__(self, in_c, mid_c, out_c, stride=1):
        super().__init__()
        # conv1: 1x1 projection to bottleneck width
        self.conv1 = nn.Conv2d(in_c, mid_c, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_c)
        
        # conv2: 3x3 spatial convolution
        self.conv2 = nn.Conv2d(mid_c, mid_c, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(mid_c)
        
        # conv3: 1x1 projection back to output width
        self.conv3 = nn.Conv2d(mid_c, out_c, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        return F.relu(out)

class StandaloneResNet50(nn.Module):
    def __init__(self, supernet, config, num_classes=61):
        super().__init__()
        self.res = config['res']
        
        # 1. Stem - Pull exact active width from the supernet instance
        stem_out = supernet.stem[0].active_out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(stem_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Stages
        self.stages = nn.ModuleList()
        curr_in_c = stem_out
        
        for i, d in enumerate(config['depth']):
            blocks = []
            # Look at the first block of the supernet stage to find our target widths
            # Note: We assume all blocks in a stage share the same mid_c/out_c
            target_block = supernet.stages[i][0]
            
            # Grabbing the exact rounded integers the supernet calculated
            mid_c = target_block.conv1.active_out_channels
            out_c = target_block.conv3.active_out_channels
            
            for j in range(d):
                stride = 2 if j == 0 and i > 0 else 1
                b_in = curr_in_c if j == 0 else out_c
                blocks.append(StaticBottleneck(b_in, mid_c, out_c, stride))
                
            self.stages.append(nn.Sequential(*blocks))
            curr_in_c = out_c
            
        self.classifier = nn.Linear(curr_in_c, num_classes)

    def forward(self, x):
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode='bilinear', align_corners=False)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        return self.classifier(x)

def extract_resnet50(supernet, config):
    supernet.to('cuda')
    supernet.eval()
    dummy_input = torch.randn(1, 3, config['res'], config['res']).to('cuda')
    with torch.no_grad():
        _ = supernet(dummy_input, config)

    # 2. Build the static model with the exact shapes found in the supernet
    student = StandaloneResNet50(supernet, config)
    student.eval()
    
    super_state = supernet.state_dict()
    student_state = student.state_dict()
    
    print(f"Slicing weights...")

    for name, param in student_state.items():
        if name in super_state:
            src = super_state[name]
            s = param.shape
            
            # --- FIXED SLICING LOGIC ---
            if len(s) == 4:   # Conv: [Out, In, H, W]
                param.copy_(src[:s[0], :s[1], :s[2], :s[3]])
            elif len(s) == 2: # Linear: [Out, In]
                param.copy_(src[:s[0], :s[1]])
            elif len(s) == 1: # BN / Bias: [Features]
                param.copy_(src[:s[0]])
            elif len(s) == 0: # Scalar (num_batches_tracked): No slicing needed!
                param.copy_(src) 
        else:
            # This helps debug if names like 'stages.0.0.conv1' match 'stages.0.conv1'
            print(f"⚠️ Warning: {name} not found in supernet state.")

    student.load_state_dict(student_state)
    return student



# =================== function to verify extraction ==================
def verify_parity(supernet, static_model, config, device="cpu"):
    # 1. Set both to eval mode (Crucial for BatchNorm/Dropout)
    supernet.eval().to(device)
    static_model.eval().to(device)
    
    # 2. Create a dummy input (matching the expected input shape)
    # Even if config['res'] is 128, you can pass 224 to test the interpolation logic
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 3. Get outputs from both
    with torch.no_grad():
        output_supernet = supernet(dummy_input, config)
        output_static = static_model(dummy_input)
    
    # 4. Compare outputs
    # atol = absolute tolerance; rtol = relative tolerance
    is_close = torch.allclose(output_supernet, output_static, atol=1e-5, rtol=1e-5)
    
    if is_close:
        print("The models match! Test is passed.")
        # Calculate the maximum difference for peace of mind
        max_diff = (output_supernet - output_static).abs().max().item()
        print(f"Max difference: {max_diff:.2e}")
    else:
        print("Models DO NOT MATCH. Ugh.")
        max_diff = (output_supernet - output_static).abs().max().item()
        print(f"Max difference: {max_diff:.2e}")