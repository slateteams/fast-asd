"""TalkNet Active Speaker Detection - Modal Deployment"""

import os
import sys

import modal

from config import image, model_volume, GPU_CONFIG, MEMORY_MB, TIMEOUT_SECONDS
from utils import get_video_path_or_download, format_results_as_json, send_callback

# Modal app configuration
app = modal.App("talknet-asd")


@app.function(
    image=image,
    volumes={"/models": model_volume},
    gpu=GPU_CONFIG,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
)
def process_video_url(
    video_url: str, 
    start_time: float = 0, 
    end_time: float = None,
    callback_url: str = None,
    job_metadata: dict = None
) -> dict:
    """
    Process video on Modal cloud from URL or local file path
    
    Args:
        video_url: URL or path to video file
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None = end of video)
        callback_url: Optional URL to POST results when complete
        job_metadata: Optional metadata dict to include in results/callback
    
    Returns:
        dict: JSON formatted results with face detection and speaking analysis
        
    Raises:
        Exception: If video processing fails
    """
    from talknet.demoTalkNet import setup, main as talknet_main
    
    # Get job ID from Modal's environment
    job_id = os.environ.get("MODAL_TASK_ID", "unknown")
    
    # Add talknet to path
    sys.path.insert(0, '/root')
    
    # Download video if URL, or validate local path
    temp_video_path, needs_cleanup = get_video_path_or_download(video_url)
    
    try:
        print(f"Processing video: {video_url} (Job ID: {job_id})")
        
        # Setup TalkNet model
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
            in_memory_threshold=3000
        )
        
        # Format results
        formatted_results = format_results_as_json(results, video_url, start_time, end_time)
        
        # Add metadata if provided
        if job_metadata:
            formatted_results["metadata"] = job_metadata
        
        # Send success callback
        if callback_url:
            send_callback(callback_url, job_id, "completed", result=formatted_results)
        
        return formatted_results
    
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error processing video: {error_msg}")
        
        # Send error callback
        if callback_url:
            send_callback(callback_url, job_id, "failed", error=error_msg)
        
        raise
    
    finally:
        # Cleanup temporary video file if needed
        if needs_cleanup and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)