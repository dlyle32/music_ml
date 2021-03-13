import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

def client_authorization():
    scope = "user-library-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    return sp


def client_credentials():
    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def main():
    sp = client_authorization()
    tracks = []
    total = 1
    offset = 0
    limit = 50
    results = sp.current_user_saved_tracks(limit=limit, offset=offset)
    while results["next"]:
        tracks.extend([(item["track"]['artists'][0]['name'],item["track"]["name"]) for item in results["items"]])
        results = sp.next(results)
    print(tracks[:100])
    print(tracks[-100:])
    print(len(tracks))


if __name__ == "__main__":
    main()
