"""
Simple token setup using device authorization flow
No redirect URI needed!
"""

import os
import requests
import time
import json
import dotenv
dotenv.load_dotenv()

def get_tokens_simple():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("❌ Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET")
        return

    # Step 1: Request device authorization
    device_url = "https://accounts.spotify.com/api/token"

    print("🎵 Simple Spotify Setup")
    print("Just paste your client credentials when prompted!")

    # For now, let's just use a pre-made token approach
    print("\n📝 Go to: https://developer.spotify.com/console/post-playlists/")
    print("1. Click 'Get Token'")
    print("2. Check 'playlist-modify-public' and 'playlist-modify-private'")
    print("3. Click 'Request Token'")
    print("4. Copy the token that appears")

    token = input("\nPaste your token here: ").strip()

    if token:
        print(f"\n✅ Add this to your .env file:")
        print(f"SPOTIFY_ACCESS_TOKEN={token}")

        # Test it
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get("https://api.spotify.com/v1/me", headers=headers)

        if response.status_code == 200:
            user = response.json()
            print(f"\n🎧 Success! Logged in as: {user['display_name']}")
        else:
            print("❌ Token test failed")

if __name__ == "__main__":
    get_tokens_simple()