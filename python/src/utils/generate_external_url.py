import os

def get_external_url(path: str) -> str:
    """Convert internal path to external URL"""
    base_url = os.environ.get('BASE_URL', 'http://localhost:8000')
    root_path = os.environ.get('ROOT_PATH', '')
    
    # Remove any double slashes
    if path.startswith('/'):
        path = path[1:]
    
    if root_path:
        if root_path.endswith('/'):
            root_path = root_path[:-1]
        return f"{base_url}{root_path}/{path}"
    else:
        return f"{base_url}/{path}"
