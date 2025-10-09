"""Video download and handling utilities"""

import os
import tempfile
import urllib.request


def download_video_from_url(url: str) -> str:
    """
    Download video from URL to temporary file
    
    Args:
        url: URL of the video to download
        
    Returns:
        str: Path to downloaded temporary file
        
    Raises:
        Exception: If download fails
    """
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    output_path = temp_file.name
    temp_file.close()
    
    print(f"Downloading video from: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        return output_path
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise Exception(f"Failed to download video: {e}")


def is_url(path: str) -> bool:
    """
    Check if path is a URL
    
    Args:
        path: String to check
        
    Returns:
        bool: True if path is a URL, False otherwise
    """
    return path.startswith(('http://', 'https://', 'ftp://'))


def get_video_path_or_download(video_input: str) -> tuple[str, bool]:
    """
    Get local video path, downloading from URL if needed
    
    Args:
        video_input: URL or local file path
        
    Returns:
        tuple: (path, needs_cleanup) where needs_cleanup indicates if file should be deleted
        
    Raises:
        FileNotFoundError: If local file doesn't exist
    """
    if is_url(video_input):
        return download_video_from_url(video_input), True
    else:
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Video file not found: {video_input}")
        return video_input, False
