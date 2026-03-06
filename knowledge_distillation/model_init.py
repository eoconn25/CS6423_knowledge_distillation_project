import torch
import torch.nn as nn
import torch.nn.functional as F

''' For our supernet model, we will use modular approach.
Need to have convolutional models whose weights we can "slice" as we grab specific paths.

For ResNet, we will be changing:
- Input size
- Depth (# convolutional blocks per stage)
- Width (# filters per conv layer)
- Expansion ratio (how many stages we have)

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
    

# this will build out each "stage" of the supernet
class CustomStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        max_mid_channels = out_channels * 6  # this is our max possible expansion, so we init with that
        
        # this encompasses a full ResNet stage - a few convolutions and normalizations
        self.conv1 = CustomConv(in_channels, max_mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(max_mid_channels)
        self.conv2 = CustomConv(max_mid_channels, max_mid_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(max_mid_channels)
        self.conv3 = CustomConv(max_mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # if a stage is excluded, we want to be able to just skip it
        self.shortcut = nn.Sequential()
        # this is our skip connection, where we want to add the original input to the output
        # if we changed the shape of our input, though, we have to scale it to match (so we can cleanly concat the two)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                CustomConv(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, expansion_ratio, is_active=True):
        if not is_active: # for Depth - if block is skipped, just return it
            return self.shortcut(x)
        
        # calculates how many channels to use based on the expansion ratio
        active_mid = int(self.conv3.active_in_channels / 6 * expansion_ratio)
        
        # set channels for internal layers to what we calculated
        self.conv1.active_out_channels = active_mid
        self.conv2.active_in_channels = active_mid
        self.conv2.active_out_channels = active_mid
        self.conv3.active_in_channels = active_mid
        
        # actual forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)  # appends our skip connection
        return F.relu(out)
    

# big class to actually orchestrate the supernet, build the blocks, etc
class Supernet(nn.Module):
    def __init__(self, width_mult_list, num_classes=122):
        super().__init__()
        # Max base channels for ResNet50
        base_channels = [64, 256, 512, 1024, 2048]
        max_w = max(width_mult_list)  # get the max width possible in search space
        
        # stem where we will take the image input and scale it up to our first layer's filters (64*max width)
        self.stem = nn.Sequential(
            nn.Conv2d(1, int(64 * max_w), kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(int(64 * max_w)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # define our stages with maximum possible blocks - follows ResNet50 architecture, scaled by width
        self.stages = nn.ModuleList([
            self.make_stage(int(64*max_w), int(256*max_w), max_blocks=4),
            self.make_stage(int(256*max_w), int(512*max_w), max_blocks=4),
            self.make_stage(int(512*max_w), int(1024*max_w), max_blocks=4),
            self.make_stage(int(1024*max_w), int(2048*max_w), max_blocks=4),
        ])
        
        # final classification layer
        self.classifier = nn.Linear(int(2048 * max_w), num_classes)

    # function to create a stage (ie assemble our blocks)
    def make_stage(self, in_c, out_c, max_blocks):
        blocks = []
        for i in range(max_blocks):
            # following resnet50 architecture, we downsample the first conv block, unless it is the first stage
            stride = 2 if i == 0 and out_c != 256 else 1
            blocks.append(CustomStage(in_c if i==0 else out_c, out_c, stride))
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
                m.active_out_channels = int(m.out_channels * current_width_mult)
                m.active_in_channels = int(m.in_channels * current_width_mult)
                
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
        x = self.classifier(x)
        return x