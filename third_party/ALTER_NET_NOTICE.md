# AlterNet attribution and modifications

The extracted code in `models/literature_alternet.py` is from Namuk Park and
Songkuk Kim's official repository for **How Do Vision Transformers Work?**
(ICLR 2022): https://github.com/xxxnell/how-do-vits-work

Pinned commit: `9571ff1370f31af2f433558fbfa7bc87a97b0bab`.
Upstream license: Apache-2.0, reproduced in `Apache-2.0.txt`.

Extracted sources: `models/alternet.py` (LocalAttention, AttentionBlockB,
StemB, AlterNet); `models/attentions.py` (Attention2d);
`models/preresnet_dnn_block.py` (Bottleneck); and
`models/classifier_block.py` (BNGAPBlock). The upstream attention module
acknowledges https://github.com/lucidrains/vit-pytorch.

Modifications: self-contained imports/helper functions; rename Bottleneck to
PreBottleneck; omit unused variants; use timm DropPath (inactive at sd=0);
register relative indices as nonpersistent buffers for device transfers.
The builder selects the author's dnn_50 layout, sets 200 classes, uses a
4x4 window and a 3x3 stride-2 stem without maxpool for native Tiny ImageNet.
No dropout, stochastic depth, cyclic shift or extra FFN has been added.

The BoTNet adaptation is separately implemented from the paper, not presented
as official author code. See `LITERATURE_BASELINES.md` for its deviations.
