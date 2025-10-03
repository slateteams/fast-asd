import os
import json
import tempfile
import sys
import urllib.request
import urllib.parse

import modal

# Modal configuration
app = modal.App("talknet-asd")

image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install_from_requirements("requirements.txt")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"])
    .env({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
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
    gpu="any",
    timeout=3600,
    memory=8192,
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
        # Check if talknet directory exists locally (mounted) or needs to be set up
        local_talknet = os.path.join(os.path.dirname(__file__), 'talknet')
        cloud_talknet = "/root/talknet"
        
        if os.path.exists(local_talknet):
            print(f"Processing video with local talknet: {video_url}")
            talknet_dir = local_talknet
        elif os.path.exists(cloud_talknet):
            print(f"Processing video on Modal cloud: {video_url}")
            talknet_dir = cloud_talknet
        else:
            raise RuntimeError("TalkNet code not found. Ensure talknet directory is available.")
        
        # Add talknet to path
        if talknet_dir not in sys.path:
            sys.path.insert(0, talknet_dir)
        
        from demoTalkNet import setup, main
        
        # Initialize model
        s, DET = setup()
        
        # Process video
        results = main(
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



def run_local(video_input, start_time=0, end_time=None):
    """Run TalkNet locally without Modal - supports URLs and local files"""
    import sys
    import os
    
    # Add talknet directory to path
    talknet_dir = os.path.join(os.path.dirname(__file__), 'talknet')
    if talknet_dir not in sys.path:
        sys.path.insert(0, talknet_dir)
    
    # Enable MPS fallback for Apple Silicon
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Get video path (download if URL)
    video_path, needs_cleanup = get_video_path_or_download(video_input)
    
    try:
        from demoTalkNet import setup, main
        
        print(f"Processing video locally: {video_input}")
        
        # Initialize model
        s, DET = setup()
        
        # Process video
        results = main(
            s=s,
            DET=DET,
            video_path=video_path,
            start_seconds=start_time,
            end_seconds=end_time,
            return_visualization=False,
            face_boxes="",
            in_memory_threshold=0
        )
        
        # Format as JSON
        return format_results_as_json(results, video_input, start_time, end_time)
    
    finally:
        # Clean up downloaded file if needed
        if needs_cleanup and os.path.exists(video_path):
            os.unlink(video_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TalkNet Active Speaker Detection",
        epilog="""
Examples:
  # Process locally (uses your GPU) - local file
  python main.py video.mp4 --local --output results.json
  
  # Process locally (uses your GPU) - from URL
  python main.py https://example.com/video.mp4 --local --output results.json
  
  # Process on Modal cloud - local file
  python main.py video.mp4 --output results.json
  
  # Process on Modal cloud - from URL
  python main.py https://example.com/video.mp4 --output results.json
  
  # Alternative: Use Modal CLI directly
  modal run main.py::process_video_url --video-url https://example.com/video.mp4 --start-time 0 --end-time 30
        """
    )
    parser.add_argument("video_input", help="Path to video file or URL (http/https)")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--local", action="store_true", help="Run locally instead of on Modal cloud")
    
    args = parser.parse_args()
    
    # Validate video input (file or URL)
    if not is_url(args.video_input) and not os.path.exists(args.video_input):
        print(f"Error: Video file not found: {args.video_input}")
        exit(1)
    
    try:
        if args.local:
            # Run locally
            results = run_local(args.video_input, args.start, args.end)
        else:
            # Run on Modal cloud
            print("Processing on Modal cloud...")
            print(f"Processing video: {args.video_input}")
            
            with app.run():
                results = process_video_url.remote(args.video_input, args.start, args.end)
        
        # Save or print results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
            
    except Exception as e:
        print(f"Error processing video: {e}")
        if not args.local:
            print("\nTry running locally: python main.py --local your_video.mp4")
        exit(1)
