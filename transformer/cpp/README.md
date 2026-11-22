# `transformer/cpp/` — C++ forward pass of the cube transformer

`Transformer.h/.cpp` run the trained transformer inside the game, with no
dependency beyond the C++ standard library. They read the text weight file that
[`../training/train.py`](../training/) exports and compute the network's output for the
last 4 camera frames; nothing is trained here.

| Step | |
|---|---|
| tokens | each 5 x 5 patch of each 20 x 20 frame, oldest frame first: 64 tokens; linear map to width 32, plus a learned position |
| 2 layers | pre-norm: x += attention(LayerNorm(x)), 2 heads, softmax(q kᵀ / √d_head) v; x += MLP(LayerNorm(x)), 32 → 64 → 32 with exact (erf) GELU |
| output | LayerNorm, mean over the 64 tokens, linear → scores for the 4 keys |

CMake compiles these files into the simulation core (`pai_core`). The check that
they give PyTorch's outputs is in [`../tests/`](../tests/).

Weight file layout: a header `transformer 1 frames height width patch model heads layers feed_forward outputs`,
then each tensor as `name count` followed by its values, in PyTorch's order
(linear weights as rows of outputs).
