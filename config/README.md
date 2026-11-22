# `config/` — Game settings and ready-made presets

Every setting of the restored simulation lives in a plain `key = value` file;
`#` starts a comment. The game reads `config/game.cfg` unless another file is
given on the command line:

```sh
./build/pai_sim                                  # config/game.cfg
./build/pai_sim config/presets/compare.cfg       # any other settings file
```

| File | Purpose |
|---|---|
| `game.cfg` | the default settings, every key with a comment |
| `plugins.cfg.in` | template for Ogre's plugin list; CMake fills in the plugin folder of the local Ogre installation |
| [`presets/`](presets/) | ready-made settings for each mode |

Paths inside a settings file (policies, scripts, recordings) are relative to the
project folder: the nearest folder above the settings file that holds `media/`.
The complete list of keys is in the [main README](../README.md#configuration);
the transformer's own settings are in [`transformer/config/`](../transformer/config/).
