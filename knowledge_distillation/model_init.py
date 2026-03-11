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
    def __init__(self, in_c, mid_c, out_c, stride=1):
        super().__init__()
        max_mid_channels = mid_c * 6  # this is our max possible expansion, so we init with that
        
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
        active_mid = round(self.conv3.active_in_channels / 6 * expansion_ratio)
        
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
    
    
# big class to actually orchestrate the supernet, build the blocks, etc
class Supernet(nn.Module):
    def __init__(self, width_mult_list, num_classes=122):
        super().__init__()
        self.max_w = max(width_mult_list)  # get the max width possible in search space
        
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
            blocks.append(CustomStage(blocks_in, int(mid_c * self.max_w), int(out_c * self.max_w), stride))
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

    
