# ACmix method reference and implementation scope

Paper: Xuran Pan et al., *On the Integration of Self-Attention and Convolution*,
CVPR 2022. https://arxiv.org/abs/2111.14556

Reference code inspected at:
https://github.com/LeapLabTHU/ACmix/tree/81dddb6dff98f5e238a7fb6ab174e256489c07fa
(`ResNet/ResNet_ImageNet.py` and `ResNet/test_bottleneck.py`).

No root license file was found in that pinned repository tree. We do not vendor
its implementation into tracked source. `models/literature_acmix.py` is an
independent implementation of the method, with the topology and numerical
operation checked against the reference. Do not label it an official author port.

Preserved method: all 16 ResNet-50 bottleneck middle operators are ACmix; four
heads; 7x7 reflection-padded sliding attention; shared Q/K/V projections; coordinate
projection and relative term; generated 3x3 grouped convolution aggregation;
two learned, input-independent scalar branch weights initialized at 0.5.

The reference assigns the return of a bias-zeroing helper (None) to its grouped
convolution bias. Our grouped convolution therefore explicitly uses no bias.
The full author ResNet constructor overwrites the standalone ACmix shift-kernel
initialization with Kaiming initialization. Our model follows that final
constructor behavior, not the standalone operator initialization.

Tiny adaptations: the same 3x3 stride-2/no-maxpool stem as our ResNet control,
200 classes, native 64x64 input, common training recipe. The whole ACmix operator
runs in FP32 under AMP. Non-reentrant activation checkpointing recomputes only
the attention branch during backward to reduce saved Unfold intermediates;
no BatchNorm or stochastic layer is recomputed. This increases training work,
does not change inference, and must be disclosed in timing comparisons.

Verification: mapped reference operator weights yield matching float64 forward
outputs and all parameter gradients (atol/rtol 1e-12), for stride 1 and 2.
Reference code used for this local check is not part of tracked source.
The authors warn that this ResNet sliding-window realization is inefficient
without specialized kernels. This baseline is for method comparison, not a
claim of competitive deployment speed.
