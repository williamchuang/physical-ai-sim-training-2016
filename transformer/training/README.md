# `transformer/training/` — PyTorch: recordings → trained, exported policy

| File | |
|---|---|
| `data.py` | reads the game's recordings (format below) and rebuilds each frame's 4-frame history, padding the start of a round as the game does |
| `model.py` | `TransformerPolicy` and `MlpPolicy` in PyTorch; exporters to the game's text formats |
| `train.py` | trains either network (AdamW, cosine schedule over a time budget), keeps the best on held-out frames, exports it |
| `make_fixture.py` | writes the data for the C++ check in `../tests/` |

```sh
pip install -r free-flight/requirements.txt       # torch, numpy
./build/pai_sim transformer/config/record.cfg      # record; close the window to stop
python transformer/training/train.py --kind transformer --out transformer/models/transformer.txt transformer/recordings/*.bin
python transformer/training/train.py --kind mlp --out transformer/models/policy-same-data.txt transformer/recordings/*.bin
python transformer/training/make_fixture.py
```

The last tenth of every recording is held out for validation.
`--init transformer/models/transformer.pt` continues from a checkpoint (used between DAgger rounds).

**Recording format** (little-endian): a header `int32 "REC1", int32 width, int32 height`,
then one record per camera frame: `int32 episode, int32 slot, int32 key (-1 if none),
uint8 network_drove, uint8 is_bot`, and width x height grey pixels as `uint8`.
