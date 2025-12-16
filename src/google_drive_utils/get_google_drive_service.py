from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from pathlib import Path
from src.config.constants import SCOPES

def get_google_drive_service():

    creds = None

    repo_root = Path(__file__).resolve().parents[2]
    token_path = repo_root / 'src' / 'token.pickle'
    credentials_path = repo_root / 'credentials.json'

    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not getattr(creds, 'valid', False):
        if creds and getattr(creds, 'expired', False) and getattr(creds, 'refresh_token', None):
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError(f"credentials.json not found at {credentials_path}. Place your credentials there or update the path in get_google_drive_service.py")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
    
        token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds, cache_discovery=False, static_discovery=True)
    return service