#!/usr/bin/env python3
"""
Script to call the deployed Modal app
"""
import modal
import json
import sys

def call_deployed_app(video_url, start_time=0, end_time=None):
    """Call the deployed TalkNet app on Modal"""
    
    # Connect to the deployed app by name
    try:
        # Use the Function.from_name method to get the deployed function
        process_video_url = modal.Function.from_name("talknet-asd", "process_video_url")
        
        print(f"Calling deployed app with video: {video_url}")
        print(f"Time range: {start_time}s to {end_time}s")
        
        # Call the deployed function
        result = process_video_url.remote(video_url, start_time, end_time)
        
        return result
    except Exception as e:
        print(f"Error connecting to deployed app: {e}")
        print("Make sure the app is deployed with: modal deploy main.py")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python call_deployed.py <video_url> [start_time] [end_time]")
        print("Example: python call_deployed.py https://example.com/video.mp4 0 30")
        sys.exit(1)
    
    video_url = sys.argv[1]
    start_time = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    end_time = float(sys.argv[3]) if len(sys.argv) > 3 else None
    
    try:
        result = call_deployed_app(video_url, start_time, end_time)
        print("\nResults:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
