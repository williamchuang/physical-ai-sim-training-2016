# `models/` — The trained 2016-style policy

| File | |
|---|---|
| `policy.txt` | the bot's network, 20 x 20 grey pixels → 32 sigmoid units → 4 keys, as text (`mlp 1 400 32 4`, then the weights) |

It learned inside the rendered game from the autopilot (`config/presets/train-autopilot.cfg`)
and is the policy every demo loads by default. Training mode keeps updating the
file named by `policy_file`, saving it every `autosave_seconds`.

Other files here are ignored by git, so local training runs do not end up in a
commit by accident. The transformer and the network retrained on its data are
in [`transformer/models/`](../transformer/models/).
