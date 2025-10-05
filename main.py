import os
import json
import tempfile
import urllib.request

import modal

# Modal configuration
app = modal.App("talknet-asd")

image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"])
    .env({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    .add_local_dir("talknet", remote_path="/root/talknet")
)

model_volume = modal.Volume.from_name("talknet-models", create_if_missing=True)


def download_video_from_url(url, output_path=None):
    """Download video from URL to local file"""
    if output_path is None:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    print(f"Downloading video from: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Video downloaded to: {output_path}")
        return output_path
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise Exception(f"Failed to download video: {e}")


def is_url(path):
    """Check if path is a URL"""
    return path.startswith(('http://', 'https://', 'ftp://'))


def get_video_path_or_download(video_input):
    """Get local video path, downloading from URL if needed"""
    if is_url(video_input):
        return download_video_from_url(video_input), True  # True = needs cleanup
    else:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Video file not found: {video_input}")
        return video_input, False  # False = no cleanup needed


def format_results_as_json(results, video_path, start_time, end_time):
    """Format TalkNet results as JSON"""
    json_output = {
        "video_info": {
            "path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "total_frames": len(results)
        },
        "frames": []
    }
    
    for frame_data in results:
        frame_info = {
            "frame_number": frame_data["frame_number"],
            "timestamp": frame_data["frame_number"] / 25.0,
            "faces": []
        }
        
        for face in frame_data["faces"]:
            face_info = {
                "track_id": face["track_id"],
                "bounding_box": {
                    "x1": face["x1"],
                    "y1": face["y1"], 
                    "x2": face["x2"],
                    "y2": face["y2"],
                    "width": face["x2"] - face["x1"],
                    "height": face["y2"] - face["y1"]
                },
                "speaking": {
                    "is_speaking": face["speaking"],
                    "confidence_score": face["raw_score"],
                    "threshold": 0.0
                }
            }
            frame_info["faces"].append(face_info)
        
        json_output["frames"].append(frame_info)
    
    return json_output


@app.function(
    image=image,
    volumes={"/models": model_volume},    
    gpu=["L4", "A10", "L40S"],
    memory=16384,
    timeout=3600,
)
def process_video_url(video_url: str, start_time: float = 0, end_time: float = None):
    """Process video on Modal cloud from URL"""
    import sys
    import os
    
    # Set up environment
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Download video from URL
    temp_video_path, needs_cleanup = get_video_path_or_download(video_url)
    
    try:
        print(f"Processing video on Modal cloud: {video_url}")
        
        # Add current directory to Python path and import talknet
        import sys
        sys.path.insert(0, '/root')
        from talknet.demoTalkNet import setup, main as talknet_main
        
        # Initialize model
        s, DET = setup()
        
        # Process video
        results = talknet_main(
            s=s,
            DET=DET,
            video_path=temp_video_path,
            start_seconds=start_time,
            end_seconds=end_time,
            return_visualization=False,
            face_boxes="",
            in_memory_threshold=0
        )
        
        # Format as JSON
        return format_results_as_json(results, video_url, start_time, end_time)
    
    finally:
        # Clean up temporary file if it was downloaded
        if needs_cleanup and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
