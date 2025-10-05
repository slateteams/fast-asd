# TalkNet Active Speaker Detection

Optimized implementation of active speaker detection using [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD).

This implementation supports:

- **Local processing** with M3 Mac GPU acceleration
- **Modal cloud** deployment for scalable processing
- **JSON output** with bounding boxes and speaking detection
- **Variable frame-rate** videos (not just 25 FPS)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run locally (free, uses your GPU)

```bash
python main.py video.mp4 --local --output results.json
```

### 3. Run on Modal cloud (production)

```bash
modal token new
python main.py video.mp4 --output results.json
```

## Usage Options

**Local Processing:**

```bash
# Basic usage (uses your M3 GPU)
python main.py video.mp4 --local --output results.json

# With time range
python main.py video.mp4 --local --start 10 --end 30 --output results.json
```

**Modal Cloud Processing:**

```bash
# Basic usage
python main.py video.mp4 --output results.json

# With time range
python main.py video.mp4 --start 10 --end 30 --output results.json

# Alternative: Modal CLI
modal run main.py::process_video --video-path video.mp4 --start-time 0 --end-time 30
```

## Output Format

Returns JSON with frame-by-frame face detection and speaking analysis:

```json
{
  "video_info": {
    "path": "video.mp4",
    "total_frames": 750
  },
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "faces": [
        {
          "track_id": 0,
          "bounding_box": {
            "x1": 100,
            "y1": 200,
            "x2": 300,
            "y2": 400,
            "width": 200,
            "height": 200
          },
          "speaking": {
            "is_speaking": true,
            "confidence_score": 0.85
          }
        }
      ]
    }
  ]
}
```
