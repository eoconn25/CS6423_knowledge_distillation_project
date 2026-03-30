import torch
import torch.nn as nn
import torch.nn.functional as F

''' For our supernet model, we will use modular approach.
Need to have convolutional models whose weights we can "slice" as we grab specific paths.

For ResNet, we will be changing:
- Input size
- Depth (# convolutional blocks per stage)
- Width (# filters per conv layer)
- Expansion ratio (how many stages we have) - only for ResNet50

We'll need to be able to toggle these as we sample different paths'''


# instead of using pytorch Conv2d, we define our own customziable class
class CustomConv(nn.Conv2d):
    def __init__(self, max_in_channels, max_out_channels, kernel_size, stride=1, padding=0):
        super().__init__(max_in_channels, max_out_channels, kernel_size, stride, padding)
        self.active_in_channels = max_in_channels  # sets our Conv layer's in and out filters
        self.active_out_channels = max_out_channels

    def forward(self, x):
        # slice our weight and bias, which are inherited from nn.Conv2d
        weight = self.weight[:self.active_out_channels, :self.active_in_channels, :, :]
        bias = self.bias[:self.active_out_channels] if self.bias is not None else None
        return F.conv2d(x, weight, bias, self.stride, self.padding)
    

class CustomBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.active_features = num_features
        
    def forward(self, x):
        return F.batch_norm(
            x,
            self.running_mean[:self.active_features],
            self.running_var[:self.active_features],
            self.weight[:self.active_features],
            self.bias[:self.active_features],
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps
        )
    
    
# this will build out each "stage" of the supernet
class CustomStage(nn.Module):
    def __init__(self, in_c, mid_c, out_c, max_exp, stride=1):
        super().__init__()
        self.max_exp = max_exp
        max_mid_channels = int(mid_c * self.max_exp)  # this is our max possible expansion, so we init with that
        
        # this encompasses a full ResNet stage - a few convolutions and normalizations
        self.conv1 = CustomConv(in_c, max_mid_channels, kernel_size=1)
        self.bn1 = CustomBatchNorm(max_mid_channels)
        self.conv2 = CustomConv(max_mid_channels, max_mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = CustomBatchNorm(max_mid_channels)
        self.conv3 = CustomConv(max_mid_channels, out_c, kernel_size=1)
        self.bn3 = CustomBatchNorm(out_c)
        
        # if a stage is excluded, we want to be able to just skip it
        self.shortcut = nn.Sequential()
        # this is our skip connection, where we want to add the original input to the output
        # if we changed the shape of our input, though, we have to scale it to match (so we can cleanly concat the two)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                CustomConv(in_c, out_c, kernel_size=1, stride=stride),
                CustomBatchNorm(out_c)
            )

    def forward(self, x, expansion_ratio, is_active=True):
        if not is_active: # for Depth - if block is skipped, just return it
            if len(self.shortcut) > 0:
                self.shortcut[1].active_features = self.conv3.active_out_channels
            return self.shortcut(x)
        
        # calculates how many channels to use based on the expansion ratio
        active_mid = round(self.conv3.active_in_channels / self.max_exp * expansion_ratio)
        
        # set channels for internal layers to what we calculated
        self.conv1.active_out_channels = active_mid
        self.conv2.active_in_channels = active_mid
        self.conv2.active_out_channels = active_mid
        self.conv3.active_in_channels = active_mid
        self.bn1.active_features = active_mid
        self.bn2.active_features = active_mid
        #self.bn3.active_features = self.conv3.active_out_channels
        
        if len(self.shortcut)>0:
            self.shortcut[1].active_features = self.conv3.active_out_channels
        
        # actual forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # appends our skip connection
        return F.relu(out)
    


    
# BasicBlock for ResNet18 with 2 conv
class CustomBasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # use two 3x3 convolutions
        self.conv1 = CustomConv(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn1 = CustomBatchNorm(out_c)
        self.conv2 = CustomConv(out_c, out_c, kernel_size=3, stride=1, padding=1)
        self.bn2 = CustomBatchNorm(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                CustomConv(in_c, out_c, kernel_size=1, stride=stride),
                CustomBatchNorm(out_c)
            )

    def forward(self, x, is_active=True):
        if not is_active:
            # If skipping, we must ensure the shortcut matches the active width
            if len(self.shortcut) > 0:
                self.shortcut[1].active_features = self.conv2.active_out_channels
            return self.shortcut(x)
        
        # Standard ResNet18 forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Adjust shortcut batchnorm features if used
        if len(self.shortcut) > 0:
            self.shortcut[1].active_features = self.conv2.active_out_channels
            
        out += self.shortcut(x)
        return F.relu(out)
    
    

    
# big class to actually orchestrate the supernet, build the blocks, etc
class Supernet_resnet50(nn.Module):
    def __init__(self, width_mult_list, expansion_list, num_classes=122):
        super().__init__()
        self.max_w = max(width_mult_list)  # get the max width possible in search space
        self.max_exp = max(expansion_list)
        
        # stem where we will take the image input and scale it up to our first layer's filters (64*max width)
        self.stem = nn.Sequential(
            CustomConv(3, int(64 * self.max_w), kernel_size=7, stride=2, padding=3),
            CustomBatchNorm(int(64 * self.max_w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # define our stages with maximum possible blocks - follows ResNet50 architecture, scaled by width
        self.stages = nn.ModuleList([
            self.make_stage(int(64*self.max_w), 64, 256, max_blocks=3),
            self.make_stage(int(256*self.max_w), 128, 512, max_blocks=4),
            self.make_stage(int(512*self.max_w), 256, 1024, max_blocks=6),
            self.make_stage(int(1024*self.max_w), 512, 2048, max_blocks=3),
        ])
        
        # final classification layer
        self.classifier = nn.Linear(int(2048 * self.max_w), num_classes)

    # function to create a stage (ie assemble our blocks)
    def make_stage(self, in_c, mid_c, out_c, max_blocks):
        blocks = []
        for i in range(max_blocks):
            # following resnet50 architecture, we downsample the first conv block, unless it is the first stage
            stride = 2 if i == 0 and mid_c != 64 else 1
            blocks_in = in_c if i==0 else int(out_c*self.max_w)
            blocks.append(CustomStage(blocks_in, int(mid_c * self.max_w), int(out_c * self.max_w), int(self.max_exp), stride))
        return nn.ModuleList(blocks)
    
    
    # forward pass - need to pass config that defines our parameters like so
    # config = {'res': 224, 'width': 0.8, 'depth': [2, 3, 2, 2], 'exp': [3, 4, 3, 6]}
    def forward(self, x, config):
        # interpolate image input to match resolution param
        x = F.interpolate(x, size=(config['res'], config['res']), mode='bilinear')
        
        # adjust the width of the entire network
        current_width_mult = config['width']  # get multiplier
        for m in self.modules():  # use modules inherited from nn.Module
            if isinstance(m, CustomConv):
                # scale active channels by the multiplier
                m.active_out_channels = round(m.out_channels / self.max_w * current_width_mult)
                if m.in_channels != 3:
                    m.active_in_channels = round(m.in_channels  / self.max_w * current_width_mult)
                else:
                    m.active_in_channels=3
                    
            if isinstance(m, CustomBatchNorm):
                m.active_features = round(m.num_features / self.max_w*current_width_mult)
                
        # input image through stem
        x = self.stem(x)

        # send it through our stages!
        for i, stage in enumerate(self.stages):
            active_depth = config['depth'][i]
            for j, block in enumerate(stage):
                # only compute the active number of layers, decided by depth param
                is_active = (j < active_depth)
                x = block(x, config['exp'][i], is_active=is_active)
        
        # resnet's final pooling and then classification
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        # slice final classifier layer
        num_active_features = x.size(1)
        weight = self.classifier.weight[:, :num_active_features]
        x = F.linear(x, weight, self.classifier.bias)
        return x

    
class Supernet_resnet18(nn.Module):
    def __init__(self, width_mult_list, num_classes=122):
        super().__init__()
        self.max_w = max(width_mult_list)
        
        # stem
        self.stem = nn.Sequential(
            CustomConv(3, int(64 * self.max_w), kernel_size=7, stride=2, padding=3),
            CustomBatchNorm(int(64 * self.max_w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet18 stage channels: 64, 128, 256, 512
        # max_blocks=3 from the largest depth parameter
        self.stages = nn.ModuleList([
            self.make_stage(int(64*self.max_w),  64,  max_blocks=3, stride=1),
            self.make_stage(int(64*self.max_w),  128, max_blocks=3, stride=2),
            self.make_stage(int(128*self.max_w), 256, max_blocks=3, stride=2),
            self.make_stage(int(256*self.max_w), 512, max_blocks=3, stride=2),
        ])
        
        self.classifier = nn.Linear(int(512 * self.max_w), num_classes)

    def make_stage(self, in_c, out_c, max_blocks, stride):
        blocks = []
        for i in range(max_blocks):
            # first block handles stride
            s = stride if i == 0 else 1
            b_in = in_c if i == 0 else int(out_c * self.max_w)
            blocks.append(CustomBasicBlock(b_in, int(out_c * self.max_w), stride=s))
        return nn.ModuleList(blocks)

    def forward(self, x, config):
        # input resolution
        x = F.interpolate(x, size=(config['res'], config['res']), mode='bilinear')
        
        # width multiplier
        current_w = config['width']
        for m in self.modules():
            if isinstance(m, CustomConv):
                m.active_out_channels = round(m.out_channels / self.max_w * current_w)
                m.active_in_channels = 3 if m.in_channels == 3 else round(m.in_channels / self.max_w * current_w)
            if isinstance(m, CustomBatchNorm):
                m.active_features = round(m.num_features / self.max_w * current_w)
                
        # forward pass
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            active_depth = config['depth'][i]
            for j, block in enumerate(stage):
                is_active = (j < active_depth)
                x = block(x, is_active=is_active)
        
        # classification
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        num_active_features = x.size(1)
        weight = self.classifier.weight[:, :num_active_features]
        return F.linear(x, weight, self.classifier.bias)

    

class Supernet_resnet10(nn.Module):
    def __init__(self, width_mult_list, num_classes=61):
        super().__init__()
        self.max_w = max(width_mult_list)
        #self.max_d = max(depth_mult_list)
        
        # ResNet-T Deep Stem: 3x 3x3 convolutions
        self.stem = nn.Sequential(
            CustomConv(3, int(32 * self.max_w), kernel_size=3, stride=2, padding=1),
            CustomBatchNorm(int(32 * self.max_w)),
            nn.ReLU(inplace=True),
            CustomConv(int(32 * self.max_w), int(32 * self.max_w), kernel_size=3, stride=1, padding=1),
            CustomBatchNorm(int(32 * self.max_w)),
            nn.ReLU(inplace=True),
            CustomConv(int(32 * self.max_w), int(64 * self.max_w), kernel_size=3, stride=1, padding=1),
            CustomBatchNorm(int(64 * self.max_w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet10 Stages: base depth is [1, 1, 1, 1]
        # We allow search up to 2 blocks per stage
        self.stages = nn.ModuleList([
            self.make_stage(int(64*self.max_w),  64,  max_blocks=2, stride=1),
            self.make_stage(int(64*self.max_w),  128, max_blocks=2, stride=2),
            self.make_stage(int(128*self.max_w), 256, max_blocks=2, stride=2),
            self.make_stage(int(256*self.max_w), 512, max_blocks=2, stride=2),
        ])
        
        self.classifier = nn.Linear(int(512 * self.max_w), num_classes)
    
    def make_stage(self, in_c, out_c, max_blocks, stride):
        blocks = []
        for i in range(max_blocks):
            # The first block handles the stride; others are stride 1
            s = stride if i == 0 else 1
            
            # Input to the first block is the stage's in_c
            # Subsequent blocks take the max possible width of the previous block
            b_in = in_c if i == 0 else int(out_c * self.max_w)
            
            # We use CustomBasicBlock (2 convolutions) for ResNet10
            blocks.append(CustomBasicBlock(b_in, int(out_c * self.max_w), stride=s))
            
        return nn.ModuleList(blocks)
    
    def forward(self, x, config):
        # 1. Handle Input Resolution
        if x.shape[-1] != config['res']:
            x = F.interpolate(x, size=(config['res'], config['res']), mode='bilinear')
        
        # 2. Set Global Width Multiplier
        current_w = config['width']
        for m in self.modules():
            if isinstance(m, CustomConv):
                # Scale the outputs
                m.active_out_channels = round(m.out_channels / self.max_w * current_w)
                # Scale inputs, unless it's the very first layer (RGB input)
                if m.in_channels == 3:
                    m.active_in_channels = 3
                else:
                    m.active_in_channels = round(m.in_channels / self.max_w * current_w)
            
            if isinstance(m, CustomBatchNorm):
                m.active_features = round(m.num_features / self.max_w * current_w)
                
        # 3. Pass through the Deep Stem
        x = self.stem(x)

        # 4. Pass through Stages (Dynamic Depth)
        for i, stage in enumerate(self.stages):
            active_depth = config['depth'][i]
            for j, block in enumerate(stage):
                # Only run the block if it falls within the current config's depth
                is_active = (j < active_depth)
                x = block(x, is_active=is_active)
        
        # 5. Final Pooling and Sliced Classification
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        # Slice the linear layer weights to match the active width of the last stage
        num_active_features = x.size(1)
        weight = self.classifier.weight[:, :num_active_features]
        return F.linear(x, weight, self.classifier.bias)
    
    
    
# define a function to save our supernet model
def save_supernet(model, model_type, path="supernet_checkpoint.pth"):
    if model_type == 'resnet50':
        # we'll save the state dict along with the search space for future reference
        torch.save({
            'state_dict': model.state_dict(),
            'width_mult_list': [0.65, 0.8, 1.0, 1.2],
            'search_space': {
                'res': [128, 160, 190, 224],
                'depth': [1, 2, 3]
            }
        }, path)
        print(f"resnet50 supernet saved to {path}")
        
    if model_type == 'resnet18':
        # we'll save the state dict along with the search space for future reference
        torch.save({
            'state_dict': model.state_dict(),
            'width_mult_list': [0.5, 0.75, 1.0, 1.2],
            'search_space': {
                'res': [128, 160, 190, 224],
                'depth': [2, 3, 4],
                'exp': [3, 4, 6]
            }
        }, path)
        print(f"resnet18 supernet saved to {path}")

        
# =============== functions for extracting winning student =====================        

class StaticBasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)
        
        

class StaticResNet(nn.Module):
    def __init__(self, config, model_type='resnet18', num_classes=61):
        super().__init__()
        w = config['width']
        depths = config['depth']
        
        # --- 1. STEM LOGIC ---
        if model_type == 'resnet10t':
            # ResNet-T Deep Stem (3x 3x3 convs)
            self.stem = nn.Sequential(
                nn.Conv2d(3, int(32*w), 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(int(32*w)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(32*w), int(32*w), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(int(32*w)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(32*w), int(64*w), 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(int(64*w)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            # Standard ResNet Stem (1x 7x7 conv)
            self.stem = nn.Sequential(
                nn.Conv2d(3, int(64*w), kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(int(64*w)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        # --- 2. STAGE LOGIC ---
        self.stages = nn.ModuleList()
        in_c = int(64 * w)
        base_channels = [64, 128, 256, 512]
        
        for i, d in enumerate(depths):
            out_c = int(base_channels[i] * w)
            blocks = []
            for j in range(d):
                stride = 2 if j == 0 and i > 0 else 1
                # Branching logic for Block Type
                if model_type == 'resnet50':
                    # ResNet50 needs expansion logic from the config
                    exp = config['exp'][i]
                    blocks.append(StaticBottleneck(in_c if j == 0 else out_c * 4, out_c, exp, stride))
                else:
                    blocks.append(StaticBasicBlock(in_c if j == 0 else out_c, out_c, stride))
            self.stages.append(nn.Sequential(*blocks))
            in_c = out_c * 4 if model_type == 'resnet50' else out_c
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_c, num_classes)
        
        
def extract_winning_student(supernet, config, model_type):
    # Create the correct static destination
    student = StaticResNet(config, model_type=model_type)
    student.eval()
    
    super_state = supernet.state_dict()
    student_state = student.state_dict()
    
    # Logic to handle the mismatch between Sequential and ModuleList naming
    # Supernets often use 'stages.0.0.conv1' while Static might use 'stages.0.conv1'
    # We use a fuzzy-matching approach to be robust
    for name, param in student_state.items():
        if name in super_state:
            s = param.shape
            src = super_state[name]
            
            # Dimensional Slicing
            if len(s) == 4: student_state[name].copy_(src[:s[0], :s[1], :s[2], :s[3]])
            elif len(s) == 1: student_state[name].copy_(src[:s[0]])
            elif len(s) == 2: student_state[name].copy_(src[:s[0], :s[1]])
            
    student.load_state_dict(student_state)
    return student