# Conformer author code

`models/conformer_author.py` derives from `conformer.py` in the authors' repository:
https://github.com/pengzhiliang/Conformer/tree/815aaad3ef5dbdfcf1e11368891416c2d7478cb1

Paper: Zhiliang Peng et al., *Conformer: Local Features Coupling Global
Representations for Visual Recognition*, ICCV 2021.

The upstream repository distributes this code under Apache License 2.0. A copy
is provided in `third_party/Apache-2.0.txt`. The upstream file has no additional
copyright header. We preserve the author implementation except for the modern
`timm.layers` import and replacing dynamic `eval` attribute access with `getattr`.

`models/literature_conformer.py` is the project adapter. It selects the author's
`Conformer_tiny_patch16` settings (ratio 1, width 384, depth 12, six heads), changes
the classifier to 200 classes, uses native 64x64 input without resizing to 224,
and wraps attention in FP32 for training stability. The original 7x7 stem and
maxpool, bidirectional FCUs, CLS token and two classification heads remain.

Training follows author `engine.py`: average the two cross-entropy losses.
Inference Top-1/Top-5 uses the sum of the two logits, as in that implementation.
Our reported evaluation NLL uses the summed logits, unlike the author's average
per-head evaluation loss; these NLL quantities must not be conflated.
The optimizer, augmentation and schedule use our common Tiny protocol, not the
paper's full ImageNet recipe. No weights are downloaded.
