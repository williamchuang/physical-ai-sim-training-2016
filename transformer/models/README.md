# `transformer/models/` — Trained cube transformer, comparison network and results

| File | |
|---|---|
| `transformer.txt` | the transformer the game runs (text weights, 20,164 parameters) |
| `transformer.pt` | the same in PyTorch, to continue training (`train.py --init`) |
| `transformer.history.json` | validation accuracy and loss during training, and the recordings used |
| `policy-same-data.txt` | the 2016 perceptron (400 → 32 → 4) trained in PyTorch on the same recordings, in the game's `mlp 1` format |
| `policy-same-data.pt`, `policy-same-data.history.json` | its PyTorch checkpoint and training record |
| `comparison.json` | coins per seed in the rendered game for the four controllers in the six-seed table |

`policy-same-data.txt` runs in the game like `models/policy.txt`:
`policy_file = transformer/models/policy-same-data.txt`, `decision_interval = 0.05`.
