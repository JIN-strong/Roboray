# HaMeR weights (offline placement manifest)

HaMeR upstream uses `fetch_demo_data.sh` to pull `hamer_demo_data.tar.gz`. The tarball is
**6.04 GB** in total, of which **3.8 GB is ViTPose's `wholebody.pth`** (only needed by the
mmpose detection chain, which this repo does not use -- see the header of
`deploy/modelenvs/build_hamer.sh`). Only `_DATA/hamer_ckpts/` (2.69 GB) and
`_DATA/data/mano_mean_params.npz` are actually needed.

## Directory layout (validated)

```
$ROBORAY_HAMER_WEIGHTS/            # e.g. /ego/weights/hamer
├── mano/
│   └── MANO_RIGHT.pkl              # registered download (see below)
└── _DATA/                          # official cache dir name, must not be changed
    ├── hamer_ckpts/
    │   ├── model_config.yaml       # load_hamer reads it from <ckpt>/../..
    │   ├── dataset_config.yaml
    │   └── checkpoints/hamer.ckpt  # 2.69 GB
    └── data/
        ├── mano_mean_params.npz
        └── mano/MANO_RIGHT.pkl     # symlink to ../../mano/MANO_RIGHT.pkl
```

**Why it must be named `_DATA` and be under CWD**: in `hamer/configs/__init__.py`,
`CACHE_DIR_HAMER = "./_DATA"` is a module constant **relative to CWD**, and `load_hamer`
uses `get_config(..., update_cachedir=True)` to build `MANO.MODEL_PATH` as
`./_DATA/data/mano`. So the running process's CWD must land on the weights directory --
`scripts/modelenvs/run_hamer.py` already does `os.chdir(weights)` itself, so callers need
not care.

## Fetch commands

```bash
# 1) MANO: register at https://mano.is.tue.mpg.de then download MANO_RIGHT.pkl (3.8 MB)
mkdir -p $W/mano && cp MANO_RIGHT.pkl $W/mano/

# 2) HaMeR weights: full tarball 6 GB (includes the unused 3.8GB ViTPose), resumable
curl -L -C - -o hamer_demo_data.tar.gz \
  https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz

# 3) extract only the needed parts (skip the 3.8GB vitpose_ckpts)
mkdir -p $W && tar -xzf hamer_demo_data.tar.gz -C $W \
  _DATA/hamer_ckpts _DATA/data

# 4) also put MANO where HaMeR expects it
mkdir -p $W/_DATA/data/mano
ln -sf ../../../mano/MANO_RIGHT.pkl $W/_DATA/data/mano/MANO_RIGHT.pkl
```

> The gdown link in `fetch_demo_data.sh` and the wget URL above are the same artifact. If
> going through an HF/Google proxy, note that `all_proxy=socks://` makes httpx fail outright;
> use `env -u all_proxy -u ALL_PROXY`.

## Correspondence (code provenance)

| File | Provenance |
| --- | --- |
| `_DATA/hamer_ckpts/checkpoints/hamer.ckpt` | `hamer/models/__init__.py::load_hamer` (`Path(ckpt).parent.parent/'model_config.yaml'`) |
| `_DATA/data/mano/MANO_RIGHT.pkl` | `hamer/configs/__init__.py::get_config(update_cachedir=True)` -> `cfg.MANO.MODEL_PATH` |
| `_DATA/data/mano_mean_params.npz` | same -> `cfg.MANO.MEAN_PARAMS` |

## License

`MANO_RIGHT.pkl` is under the MPI license and **must not be committed/re-distributed**. The
repo's `.gitignore` already excludes `MANO*.pkl` and `mano/`; every user must register at
https://mano.is.tue.mpg.de and download it individually.
