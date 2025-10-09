# TalkNet Active Speaker Detection

Cloud-based active speaker detection using [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD), deployed on [Modal](https://modal.com).

Features:

- **Modal cloud processing** with GPU acceleration (L4, A10, or L40S)
- **JSON output** with bounding boxes and speaking detection
- **URL support** - process videos from URLs or local files
- **Flexible time ranges** - process specific segments
- **Async processing** with callback support for AWS Lambda integration
- **Job tracking** - submit jobs and get results via callback URL

## Quick Start

### 1. Setup

```bash
pip install modal
modal token new
```

### 2. Process a video

```bash
modal run main.py::process_video_url --video-url video.mp4
```

## Usage

```bash
# Basic usage
modal run main.py::process_video_url --video-url video.mp4

# Process specific time range
modal run main.py::process_video_url --video-url video.mp4 --start-time 10 --end-time 30

# Process from URL
modal run main.py::process_video_url --video-url https://example.com/video.mp4

# Deploy as a persistent endpoint
modal deploy main.py
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
