import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

## Largely pulled from https://github.com/markkohdev/spotify-api-starter/

def client_authorization():
    scope = "user-library-read"
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
    return sp

def client_credentials():
    auth_manager = SpotifyClientCredentials()
    sp = spotipy.Spotify(auth_manager=auth_manager)
    return sp

def search_track(spotify):
    """
    This demo function will allow the user to search a song title and pick the song from a list in order to fetch
    the audio features/analysis of it
    :param spotify: An basic-authenticated spotipy client
    """
    keep_searching = True
    selected_track = None

    # Initialize Spotipy
    # spotify = client_authorization()

    # We want to make sure the search is correct
    while keep_searching:
        search_term = input('\nWhat song would you like to search: ')

        # Search spotify
        results = spotify.search(search_term)
        tracks = results.get('tracks', {}).get('items', [])

        if len(tracks) == 0:
            print_header('No results found for "{}"'.format(search_term))
        else:
            # Print the tracks
            print_header('Search results for "{}"'.format(search_term))
            for i, track in enumerate(tracks):
                print('  {}) {}'.format(i + 1, track_string(track)))

        # Prompt the user for a track number, "s", or "c"
        track_choice = input('\nChoose a track #, "s" to search again, or "c" to cancel: ')
        try:
            # Convert the input into an int and set the selected track
            track_index = int(track_choice) - 1
            selected_track = tracks[track_index]
            keep_searching = False
        except (ValueError, IndexError):
            # We didn't get a number.  If the user didn't say 'retry', then exit.
            if track_choice != 's':
                # Either invalid input or cancel
                if track_choice != 'c':
                    print('Error: Invalid input.')
                keep_searching = False

    # Quit if we don't have a selected track
    if selected_track is None:
        return

    # Request the features for this track from the spotify API
    # get_audio_features(spotify, [selected_track])

    return [selected_track]

def print_header(message, length=30):
    """
    Given a message, print it with a buncha stars all header-like
    :param message: The message you want to print
    :param length: The number of stars you want to surround it
    """
    print('\n' + ('*' * length))
    print(message)
    print('*' * length)


def choose_tracks(tracks):
    """
    Given a list of tracks, list them on the console and let the user choose a
    selection of them.
    :return: A list of selected track objects
    """
    for i, track in enumerate(tracks):
        print('  {}) {}'.format(i + 1, track_string(track)))

    # Choose some tracks
    track_choices = input('\nChoose some tracks (e.g 1,4,5,6,10): ')

    # Turn the input into a list of integers
    try:
        track_choice_indexes = [int(choice.strip()) for choice in track_choices.split(',')]
    except ValueError as e:
        print('Error: Invalid input.')
        return []

    # Grab the tracks from our track list and return them
    selected_tracks = [tracks[index - 1] for index in track_choice_indexes]
    return selected_tracks

def track_string(track):
    """
    Given a track, return a string describing the track:
    Track Name - Artist1, Artist2, etc...
    :param track:
    :return: A string describing the track
    """
    track_name = track.get('name')
    artist_names = ', '.join([artist.get('name') for artist in track.get('artists', [])])
    return '{} - {}'.format(track_name, artist_names)