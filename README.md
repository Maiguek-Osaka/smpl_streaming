# edge2sonic — SMPL Motion Streaming for SONIC

Streams SMPL human motion data over **ZMQ** to the SONIC whole-body controller running on a Unitree G1 robot. Motion is post-processed by [Kimodo](https://github.com/nv-tlabs/kimodo) and published as a real-time pose stream at a configurable FPS.

```
SMPL file (.npz / .pkl / .pt)
        │
        ▼
  edge_to_sonic.py          ← loads & post-processes motion
        │  ZMQ PUB (tcp://host:port)
        ▼
  SONIC controller           ← receives pose frames and drives the robot
```

---

## Installation

### 1. Clone (with submodules)

```bash
git clone --recurse-submodules https://github.com/Maiguek/edge2sonic_streaming.git
cd edge2sonic_streaming
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate edge2sonic
```

### 3. Install local packages

```bash
pip install -e kimodo/
pip install -e gear_sonic/
```

That's it. No other dependencies are required.

---

## Running the streamer

```bash
python edge_to_sonic.py --input data/test_tango.pkl --fps 50
```

The script will post-process the motion and start publishing pose frames on `tcp://127.0.0.1:5556`.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | SMPL file: `.npz` (AMASS), `.pkl` (EDGE), or `.pt` |
| `--fps` | `50.0` | Source framerate of the motion file |
| `--host` | `127.0.0.1` | ZMQ publisher host |
| `--port` | `5556` | ZMQ publisher port |
| `--no_legs` | off | Zero out leg joints (upper-body only) |
| `--disable_ytoz` | off | Skip Y→Z up-axis conversion |
| `--post_process` | `True` | Run Kimodo motion post-processing |

### Example: stream to a remote machine

```bash
python edge_to_sonic.py --input data/test_song.pkl --fps 50 --host 0.0.0.0 --port 5556
```

The SONIC controller (or any ZMQ SUB socket) should subscribe to `tcp://<streamer-ip>:5556` and listen for `pose` topic messages.

---

## Supported input formats

| Extension | Source |
|---|---|
| `.npz` | AMASS dataset |
| `.pkl` | EDGE-generated motion |
| `.pt` | PyTorch SMPL parameter dict |

---

## Project structure

```
edge2sonic_streaming/
├── edge_to_sonic.py        # main streaming script
├── utils/                  # rotation math, ZMQ packing, IK solver
├── gear_sonic/             # robot model & teleop utilities (Unitree G1)
├── kimodo/                 # motion post-processing (git submodule)
├── data/                   # sample SMPL files for testing
└── environment.yml         # minimal conda environment
```
