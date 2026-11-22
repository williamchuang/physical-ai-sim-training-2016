# `transformer/free-flight/` — Transformer policy for the free-flight bots

| File | |
|---|---|
| `flight_transformer.py` | `FlightTransformer`: each of a bot's last 8 observations is one token; 2 pre-norm attention layers (2 heads, width 32, GELU MLP); the newest token gives the thrust |
| `test_flight_transformer.py` | its answer depends on its memory; backpropagation through the simulator reaches every weight; the shipped policy loads and flies without touching the surface |
| [`models/`](models/) | the trained policy and its records |

It plugs into the free-flight code through `make_policy("transformer")` in
`free-flight/policy.py`, so training, evaluation and the viewer work unchanged:

```sh
cd free-flight
python train.py --kind transformer --episodes 100 --out ../transformer/free-flight/models/flight_transformer.pt
python evaluate.py --policy ../transformer/free-flight/models/flight_transformer.pt
python viewer.py demo --policy ../transformer/free-flight/models/flight_transformer.pt --bots 3
cd .. && python -m pytest transformer/free-flight
```

The policy keeps its short memory itself: `reset()` when a new world starts,
`detach()` wherever the simulator is detached for truncated backpropagation.
