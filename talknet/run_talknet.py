#!/usr/bin/env python3
"""
Simple script to run TalkNet on a video file
Usage: python run_talknet.py path/to/video.mp4
"""

import sys
import os
import json

# Enable MPS fallback for unsupported operations on Apple Silicon
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from demoTalkNet import setup, main

def run_talknet(video_path, start_time=0, end_time=None, return_viz=False, output_json=False):
    """
    Run TalkNet on a video file
    
    Args:
        video_path: Path to the video file
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None for full video)
        return_viz: Whether to return visualization video (default: False)
        output_json: Whether to output results as JSON (default: False)
    
    Returns:
        List of frame results with face detections and speaking scores
    """
    
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Initializing TalkNet model...")
    # Initialize model and face detector
    s, DET = setup()
    
    print(f"Processing video: {video_path}")
    # Run TalkNet
    results = main(
        s=s,
        DET=DET,
        video_path=video_path,
        start_seconds=start_time,
        end_seconds=end_time,
        return_visualization=return_viz,
        face_boxes="",
        in_memory_threshold=0
    )
    
    if output_json:
        return format_results_as_json(results, video_path, start_time, end_time)
    
    return results

def format_results_as_json(results, video_path, start_time, end_time):
    """
    Format TalkNet results as JSON with proper structure
    """
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
            "timestamp": frame_data["frame_number"] / 25.0,  # Assuming 25 FPS processing
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_talknet.py <video_path> [start_time] [end_time] [return_viz] [--json] [--output file.json]")
        print("Example: python run_talknet.py video.mp4")
        print("Example: python run_talknet.py video.mp4 10 30 True")
        print("Example: python run_talknet.py video.mp4 --json")
        print("Example: python run_talknet.py video.mp4 --json --output results.json")
        sys.exit(1)
    
    video_path = sys.argv[1]
    start_time = 0
    end_time = None
    return_viz = False
    output_json = False
    output_file = None
    
    # Parse arguments
    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--json':
            output_json = True
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 1
        elif arg.lower() == 'true':
            return_viz = True
        elif arg.replace('.', '').replace('-', '').isdigit():
            if start_time == 0:
                start_time = float(arg)
            elif end_time is None:
                end_time = float(arg)
        i += 1
    
    try:
        results = run_talknet(video_path, start_time, end_time, return_viz, output_json)
        
        if output_json:
            if output_file:
                # Save to file
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"JSON results saved to: {output_file}")
            else:
                # Print to stdout
                print(json.dumps(results, indent=2))
        else:
            print(f"\nProcessing complete!")
            print(f"Found {len(results)} frames with face detections")
            
            # Print summary of first few frames
            for i, frame in enumerate(results[:5]):
                print(f"Frame {frame['frame_number']}: {len(frame['faces'])} faces detected")
                for j, face in enumerate(frame['faces']):
                    speaking = "SPEAKING" if face['speaking'] else "NOT SPEAKING"
                    print(f"  Face {j}: {speaking} (score: {face['raw_score']:.2f})")
            
            if len(results) > 5:
                print(f"... and {len(results) - 5} more frames")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
