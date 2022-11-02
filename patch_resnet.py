import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchNorm(nn.Module):
    def __init__(self, num_features=None, num_patches=None, p_sz=None, batch_or_patch='patch'):
        super().__init__()
        self.batch_or_patch = batch_or_patch
        if batch_or_patch == 'patch':
            self.norm = nn.LayerNorm([num_patches, p_sz, p_sz])
            self.num_patches = num_patches
        else:
            self.norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        if self.batch_or_patch == 'batch':
            return self.norm(x)

        A, C, P, _ = x.shape
        L = self.num_patches
        B = A // L
        x = x.reshape(B, L, C, P, P)
        x = torch.einsum('blcpq->bclpq', x)
        x = self.norm(x)
        x = torch.einsum('bclpq->blcpq', x)
        return x.reshape(A, C, P, P)

class PatchResNet(nn.Module):
    def __init__(self, out_chans, p_sz=16, input_sz=224, normtype='batch'):
        super().__init__()
        self.out_chans = out_chans
        self.p_sz = p_sz
        self.num_patches = (input_sz // self.p_sz) ** 2
        self.conv1 = conv(3, 32, 3)
        self.bn1 = PatchNorm(32, self.num_patches, p_sz, batch_or_patch=normtype)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = Block_3_1(32, self.num_patches, p_sz//2, normtype=normtype)
        self.conv2 = conv(32, 64, 1)
        self.bn2 = PatchNorm(64, self.num_patches, p_sz//4, batch_or_patch=normtype)
        self.block2 = Block_3_1(64, self.num_patches, p_sz//4, normtype=normtype)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, out_chans)


    def patchify(self, imgs):
        """
        imgs: (B, C, H, H)
        x: (B*(H//P)**2, C, P, P)
        """
        P = self.p_sz
        B, C, H, W = imgs.shape
        assert H == W and H % P == 0

        h = H // P
        x = imgs.reshape(B, C, h, P, h, P)
        x = torch.einsum('bchpwq->bhwcpq', x)
        x = x.reshape(B*h*h, C, P, P)
        return x


    def forward(self, x):
        x = self.patchify(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.bn2(x)
   
        x = self.block2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B*h*h, C*P*P]
        x = self.fc(x)  # [B*h*h, out_chans]

        x = x.view(-1, self.num_patches, self.out_chans)

        return x


class Block_3_1(nn.Module):
    def __init__(self, chans, num_patches, p_sz, normtype):
        super().__init__()
        self.conv1 = conv(chans, chans, 3)
        self.bn1 = PatchNorm(chans, num_patches, p_sz, batch_or_patch=normtype)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(chans, chans, 1)
        self.bn2 = PatchNorm(chans, num_patches, p_sz, batch_or_patch=normtype)
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out




def conv(in_chans, out_chans=None, k_sz=3):
    if out_chans is None:
        out_chans = in_chans
    return nn.Conv2d(
        in_chans,
        out_chans,
        kernel_size=k_sz,
        stride=1,
        bias=False,
        padding=k_sz//2
    )




# class BasicBlock(nn.Module):
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x: Tensor) -> Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out



# class PatchResNet(nn.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         num_classes: int = 1000,
#         zero_init_residual: bool = False,
#         groups: int = 1,
#         width_per_group: int = 64,
#         replace_stride_with_dilation: Optional[List[bool]] = None,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         _log_api_usage_once(self)
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             # each element in the tuple indicates if we should replace
#             # the 2x2 stride with a dilated convolution instead
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError(
#                 "replace_stride_with_dilation should be None "
#                 f"or a 3-element tuple, got {replace_stride_with_dilation}"
#             )
#         self.groups = groups
#         self.base_width = width_per_group
#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         dilate: bool = False,
#     ) -> nn.Sequential:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 norm_layer(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(
#                 self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
#             )
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(
#                     self.inplanes,
#                     planes,
#                     groups=self.groups,
#                     base_width=self.base_width,
#                     dilation=self.dilation,
#                     norm_layer=norm_layer,
#                 )
#             )

#         return nn.Sequential(*layers)

#     def _forward_impl(self, x: Tensor) -> Tensor:
#         # See note [TorchScript super()]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         # x = self.layer2(x)
#         # x = self.layer3(x)
#         # x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)


# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:
#     if weights is not None:
#         _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

#     model = ResNet(block, layers, **kwargs)

#     if weights is not None:
#         model.load_state_dict(weights.get_state_dict(progress=progress))

#     return model


# _COMMON_META = {
#     "min_size": (1, 1),
#     "categories": _IMAGENET_CATEGORIES,
# }


 