"""
pwkd.py  —  Pruning While Knowledge Distillation
Adapted from Wang et al. (2025) for ResNet classification on RadImageNet.

Public API:
    PWKDLoss                            — CE + KD + sparsity loss
    make_aux_fn(teacher)                — produces teacher logits each batch
    finalise_student(student, loss, loader, device)  — applies pruning post-training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Variance Average Pooling  (Wang et al. Eq. 3) ─────────────────────────

class VarianceAveragePool(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        return ((x - mean) ** 2).mean(dim=(2, 3), keepdim=True)


# ── 2. Backward-differentiable gating  (Wang et al. Eqs. 4-8) ────────────────

class _DiffGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta):
        ctx.save_for_backward(x)
        ctx.theta = theta
        return (x >= theta).float()

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        theta = ctx.theta
        lo, hi = theta - 1/3, theta + 1/3
        grad_A = torch.zeros_like(x)
        m = (x >= lo) & (x < hi)
        grad_A[m] = (9/4) * (1 - 9 * x[m] ** 2)
        return grad_out * grad_A, None

def differentiable_gate(x, theta):
    return _DiffGate.apply(x, theta)


# ── 3. Auto-Pruning Module  (Wang et al. Section 3.2) ────────────────────────

class AutoPruningModule(nn.Module):
    def __init__(self, num_channels, pruning_ratio):
        super().__init__()
        self.pruning_ratio = pruning_ratio
        self.C      = num_channels
        self.vap    = VarianceAveragePool()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.fc     = nn.Linear(num_channels, num_channels)

    def forward(self, feature_map):
        v = self.vap(feature_map).squeeze(-1).squeeze(-1)
        v = v.mean(0, keepdim=True).unsqueeze(1)
        v = self.conv1d(v).squeeze(1)
        S = self.fc(v).squeeze(0)
        k = max(1, min(int(math.floor(self.pruning_ratio * self.C)), self.C - 1))
        k = int(math.floor(self.pruning_ratio * self.C))
        k = min(k, self.C - 1)

        if k == 0:
            return torch.ones_like(S)

        sorted_S, _ = torch.sort(S)
        theta = 0.5 * (sorted_S[k - 1].item() + sorted_S[k].item())
        return differentiable_gate(S, theta)   # (C,)


# ── 4. Feature Alignment Module  (stable MVRM analogue) ──────────────────────

class FeatureAlignModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std  = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-5)
        return self.align((x - mean) / std)


# ── 5. PWKD Loss ──────────────────────────────────────────────────────────────

class PWKDLoss(nn.Module):
    HOOK_LAYERS = ['layer2', 'layer3', 'layer4']

    def __init__(self, student, teacher, pruning_ratio,
                 teacher_channels, student_channels,
                 class_weights=None, lam=0.2, kd_temp=4.0, sparse_weight=1e-4):
        super().__init__()
        self.lam           = lam
        self.kd_temp       = kd_temp
        self.sparse_weight = sparse_weight
        self.ce            = nn.CrossEntropyLoss(weight=class_weights)
        same_arch          = (teacher_channels == student_channels)

        self.pruning_modules = nn.ModuleDict()
        self._conv_map       = {}
        for name, mod in student.named_modules():
            if isinstance(mod, nn.Conv2d):
                safe = name.replace('.', '_')
                self.pruning_modules[safe] = AutoPruningModule(
                    mod.out_channels, pruning_ratio)
                self._conv_map[safe] = mod

        self.teacher_proj = nn.ModuleDict()
        self.teacher_mvrm = nn.ModuleDict()
        self.student_mvrm = nn.ModuleDict()
        for layer in self.HOOK_LAYERS:
            s_ch = student_channels[layer]
            t_ch = teacher_channels[layer]
            self.teacher_proj[layer] = nn.Identity() if same_arch else \
                                       nn.Conv2d(t_ch, s_ch, 1, bias=False)
            self.teacher_mvrm[layer] = FeatureAlignModule(s_ch)
            self.student_mvrm[layer] = FeatureAlignModule(s_ch)

        self._t_feats      = {}
        self._s_feats      = {}
        self._s_conv_feats = {}
        self._hooks        = []
        self._register_hooks(student, teacher)

    def _register_hooks(self, student, teacher):
        def make_hook(store, key):
            def h(_, __, out): store[key] = out
            return h
        for name, mod in teacher.named_modules():
            if name in self.HOOK_LAYERS:
                self._hooks.append(
                    mod.register_forward_hook(make_hook(self._t_feats, name)))
        for name, mod in student.named_modules():
            if name in self.HOOK_LAYERS:
                self._hooks.append(
                    mod.register_forward_hook(make_hook(self._s_feats, name)))
        for name, mod in student.named_modules():
            if isinstance(mod, nn.Conv2d):
                safe = name.replace('.', '_')
                self._hooks.append(
                    mod.register_forward_hook(make_hook(self._s_conv_feats, safe)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, student_outputs, labels, aux=None):
        l_ce = self.ce(student_outputs, labels)
        if aux is None:
            return l_ce

        T    = self.kd_temp
        l_ts = F.kl_div(
            F.log_softmax(student_outputs / T, dim=1),
            F.softmax(aux / T, dim=1),
            reduction='batchmean') * (T ** 2)

        l_mv = torch.tensor(0.0, device=student_outputs.device)
        for layer in self.HOOK_LAYERS:
            if layer not in self._t_feats or layer not in self._s_feats:
                continue
            t_feat = self._t_feats[layer]
            s_feat = self._s_feats[layer]
            if t_feat.shape[2:] != s_feat.shape[2:]:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[2:],
                                       mode='bilinear', align_corners=False)
            mv_t = self.teacher_mvrm[layer](self.teacher_proj[layer](t_feat))
            mv_s = self.student_mvrm[layer](s_feat)
            l_mv = l_mv + F.l1_loss(mv_t, mv_s)

        l_sparse = torch.tensor(0.0, device=student_outputs.device)
        for safe, apm in self.pruning_modules.items():
            feat = self._s_conv_feats.get(safe)
            if feat is None:
                continue
            mask     = apm(feat)
            conv     = self._conv_map[safe]
            ch_norms = conv.weight.abs().sum(dim=(1, 2, 3))
            l_sparse = l_sparse + ((1.0 - mask) * ch_norms).mean()

        return l_ce + self.lam * (l_ts + l_mv) + self.sparse_weight * l_sparse


# ── 6. Public helpers ─────────────────────────────────────────────────────────

def make_aux_fn(teacher):
    def aux_forward_fn(images):
        with torch.no_grad():
            return teacher(images)
    return aux_forward_fn



# Layers whose output feeds into a residual addition — unsafe to prune.
# Only conv1 inside each BasicBlock is safe (its output feeds conv2, not the skip).
_SKIP_CONNECTED = {
    'conv1',
    'layer1.0.conv2', 'layer1.1.conv2',
    'layer2.0.conv2', 'layer2.1.conv2',
    'layer3.0.conv2', 'layer3.1.conv2',
    'layer4.0.conv2', 'layer4.1.conv2',
    'layer1.0.downsample.0', 'layer2.0.downsample.0',
    'layer3.0.downsample.0', 'layer4.0.downsample.0',
}


def finalise_student(student, pwkd_loss, data_loader, device):
    """
    After train_all():
    1. Run one batch to populate _s_conv_feats.
    2. For each conv1 in residual blocks (safe — output does NOT feed a skip
       connection), zero pruned output channels and the matching input channels
       of the paired conv2. Skip-connected layers are left untouched entirely.
    3. Remove hooks and return the student.
    """
    student.eval()
    images, _ = next(iter(data_loader))
    with torch.no_grad():
        student(images.to(device))

    conv_list    = [(n, m) for n, m in student.named_modules()
                    if isinstance(m, nn.Conv2d)]
    conv_by_name = {n: m for n, m in conv_list}

    # Map each conv to the next conv in the list (its paired conv2)
    paired = {}
    for i, (name, _) in enumerate(conv_list):
        if i + 1 < len(conv_list):
            # paired[name] = conv_list[i + 1][0]
            paired[name] = name.replace('conv1', 'conv2')

    total_pruned   = 0
    total_channels = 0

    for name, conv in conv_list:
        if name in _SKIP_CONNECTED:
            continue
        if not name.endswith('conv1'):
            continue

        safe = name.replace('.', '_')
        feat = pwkd_loss._s_conv_feats.get(safe)
        if feat is None:
            continue
        if safe not in pwkd_loss.pruning_modules:
            continue

        with torch.no_grad():
            mask = pwkd_loss.pruning_modules[safe](feat)
        prune_idx = (mask < 0.5).nonzero(as_tuple=True)[0]

        total_channels += conv.out_channels
        if len(prune_idx) == 0:
            continue

        with torch.no_grad():
            conv.weight.data[prune_idx] = 0.0
            if conv.bias is not None:
                conv.bias.data[prune_idx] = 0.0

        next_name = paired.get(name)
        if next_name and next_name in conv_by_name:
            nc = conv_by_name[next_name]
            if nc.in_channels == conv.out_channels:
                with torch.no_grad():
                    nc.weight.data[:, prune_idx] = 0.0

        total_pruned += len(prune_idx)

    pct = 100 * total_pruned / max(total_channels, 1)
    print(f'Finalised: {total_pruned}/{total_channels} conv1 channels zeroed ({pct:.1f}%)')
    pwkd_loss.remove_hooks()
    student.eval()
    return student