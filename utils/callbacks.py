"""Callback handling utilities"""


def send_callback(callback_url: str, job_id: str, status: str, result: dict = None, error: str = None) -> None:
    """
    Send results to callback URL via HTTP POST
    
    Args:
        callback_url: URL to send the callback to
        job_id: Unique identifier for the job
        status: Status of the job ("completed" or "failed")
        result: Optional result data to include
        error: Optional error message to include
        
    Note:
        Callback failures are logged but don't raise exceptions to avoid failing the job
    """
    if not callback_url:
        return
    
    import requests
    
    payload = {
        "job_id": job_id,
        "status": status,
    }
    
    if result:
        payload["result"] = result
    if error:
        payload["error"] = error
    
    try:
        print(f"Sending callback to: {callback_url}")
        response = requests.post(
            callback_url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        print(f"✓ Callback sent successfully: {response.status_code}")
    except Exception as e:
        print(f"✗ Failed to send callback: {e}")
        # Don't raise - callback failure shouldn't fail the job
