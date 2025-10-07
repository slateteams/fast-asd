"""Result formatting utilities"""


def format_results_as_json(results: list, video_path: str, start_time: float, end_time: float) -> dict:
    """
    Format TalkNet results as structured JSON
    
    Args:
        results: List of frame results from TalkNet
        video_path: Path or URL of the processed video
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        dict: Formatted JSON structure with video info and frame-by-frame results
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
            "timestamp": frame_data["frame_number"] / 25.0,  # Assuming 25 FPS
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
