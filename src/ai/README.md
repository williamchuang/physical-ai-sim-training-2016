# `src/ai/` — The perceptron and imitation learning

The learning code of 2016, written from scratch without a neural-network library.

| File | |
|---|---|
| `Mlp.h/.cpp` | the two-layer perceptron: input → sigmoid hidden layer → softmax over 4 keys; cross-entropy, minibatch gradient descent, text save and load |
| `ImitationLearner.h/.cpp` | a ring buffer of (camera image, key) demonstrations, random minibatches drawn with Floyd's sampling, atomic policy saving |

Backpropagation is checked against finite differences in `tests/ai_tests.cpp`.
The transformer's forward pass, run from PyTorch weights, is in
[`transformer/cpp/`](../../transformer/cpp/).
