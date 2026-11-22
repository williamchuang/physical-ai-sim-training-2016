# `media/` — Materials and font for the Ogre renderer

Ogre loads every resource under this folder at start-up. The track, cube and
coin shapes are not files: the game generates them from the same formula the
physics uses, so what is drawn and what is simulated always agree.

| Folder | |
|---|---|
| [`materials/`](materials/) | colours and lighting of the track, cubes, coins, sky and panels |
| [`fonts/`](fonts/) | the HUD and console font |

The game also finds its project folder through this one: it is the nearest
folder above the settings file that holds `media/`.
