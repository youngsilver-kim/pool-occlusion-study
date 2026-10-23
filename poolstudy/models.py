from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + self.shortcut(x))

class CIFARResNet(nn.Module):
    """6n+2 nominal depth, projection shortcuts; 32x32 -> 8x8.

    This is a CIFAR ResNet with option-B shortcuts, not torchvision ResNet-18.
    """
    def __init__(self, depth=20, width=16):
        super().__init__()
        n = (depth - 2) // 6
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
        blocks, cin = [], width
        for stage, cout in enumerate((width, width * 2, width * 4)):
            for i in range(n):
                blocks.append(Block(cin, cout, 2 if stage and i == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.channels = cin
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.blocks(self.stem(x))

def normalized_entropy(weights):
    n = weights.shape[-1]
    return -(weights * weights.clamp_min(1e-12).log()).sum(-1) / math.log(n)

class PoolHead(nn.Module):
    def __init__(self, channels, method, cfg):
        super().__init__()
        self.method, self.cfg = method, cfg
        if method in ("attention", "attention_entropy", "attention_floor", "mixed_attention"):
            # A scalar bias would cancel under softmax, so it is omitted.
            self.score = nn.Conv2d(channels, 1, 1, bias=False)
            nn.init.normal_(self.score.weight, std=0.01)
        if method == "gem":
            self.raw_p = nn.Parameter(torch.tensor(math.log(math.expm1(2.0))))

    @property
    def p(self):
        # p >= 1; initialized at 3. Log-space computation avoids x**p overflow.
        return 1.0 + F.softplus(self.raw_p)

    def forward(self, x, diagnostics=False):
        flat = x.flatten(2)
        weights = None
        penalty = x.new_zeros(())
        if self.method == "gap":
            z = flat.mean(-1)
            if diagnostics:
                weights = torch.full((x.shape[0], flat.shape[-1]), 1 / flat.shape[-1], device=x.device)
        elif self.method == "max":
            z = flat.amax(-1)
        elif self.method == "gem":
            p = self.p
            z = ((torch.logsumexp(flat.clamp_min(1e-6).log() * p, dim=-1) - math.log(flat.shape[-1])) / p).exp()
        else:
            a = self.score(x).flatten(1).softmax(-1)
            weights = a
            if self.training and self.method in ("attention_entropy", "attention_floor"):
                h = normalized_entropy(a)
                if self.method == "attention_entropy":
                    # Normalized KL(a || uniform) = 1 - normalized entropy.
                    penalty = self.cfg["entropy_lambda"] * (1 - h).mean()
                else:
                    penalty = self.cfg["floor_lambda"] * F.relu(self.cfg["entropy_floor"] - h).square().mean()
            if self.method == "mixed_attention":
                alpha = self.cfg["mix_alpha"]
                weights = alpha * a + (1 - alpha) / flat.shape[-1]
            z = (flat * weights[:, None, :]).sum(-1)
        return z, penalty, weights

class Classifier(nn.Module):
    def __init__(self, cfg, method, seed):
        super().__init__()
        # Isolated RNG scopes give every pooling method identical initial
        # backbone and classifier, regardless of head parameter count.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.backbone = CIFARResNet(int(cfg["backbone"].replace("resnet", "")), cfg["width"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 10001)
            self.pool = PoolHead(self.backbone.channels, method, cfg)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 20001)
            self.fc = nn.Linear(self.backbone.channels, 10 if cfg["dataset"] == "cifar10" else 100)

    def forward(self, x, diagnostics=False):
        z, penalty, weights = self.pool(self.backbone(x), diagnostics)
        logits = self.fc(z)
        return (logits, penalty, weights) if diagnostics else (logits, penalty)

def count_macs(model, device):
    """Count Conv2d and Linear MACs; functional pooling/BN/ReLU are excluded."""
    total = [0]
    handles = []
    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            total[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        elif isinstance(module, nn.Linear):
            total[0] += output.numel() * module.in_features
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, 32, 32, device=device))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return total[0]
