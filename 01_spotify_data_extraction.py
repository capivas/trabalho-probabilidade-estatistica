import os
import spotipy
import pandas as pandas

from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from data_preparation import get_dataset_path

load_dotenv()

auth_manager = SpotifyOAuth(
	client_id = os.getenv("SPOTIFY_CLIENT_ID"),
	client_secret = os.getenv("SPOTIFY_CLIENT_SECRET"),
	redirect_uri="http://127.0.0.1:8888/callback",
	scope="playlist-read-private playlist-read-collaborative"
)
sp_client = spotipy.Spotify(auth_manager=auth_manager)

annual_playlists = {
	2010: "5ZS9pYgTmybr3ci7LdZVRW",
	2011: "2eRQ2GefRVucxxdPfFw92w",
	2012: "5fkMnPqY0ilax2ZEAyhvQl",
	2013: "7kPgVGmUNL8ZPLLXjyxsYJ",
	2014: "4R0umj6e3u1JY6XD4gZZnt",
	2015: "01DHv1EhNLmLZOZ7Xft2kd",
	2016: "7yL74sMdtghrje0PngyyeB",
	2017: "2C2mTl6wW8cUnwEl29whXr",
	2018: "3NogZcZzzFF3Bgmi2KMDvT",
	2019: "6cCdvEVpN9RKWTgwz3UjhH",
	2020: "6ho6OCypoafwr3HEqbQXpE",
	2021: "25vaJELb9hxFqSsZn0YgiN",
	2022: "3G3wk1cUcvhQcQkoAuQawZ",
	2023: "7E0xrKqCNtVUrKlFN9OmRh",
	2024: "1YnHrEdgKANonoC6xejEPo",
	2025: "25dBkFPHERTFYP0e7ue0iL"
}

music_data = []
track_id_counter = 0

FIELDS = "items(item(id,name,duration_ms,explicit,album(release_date,album_type),artists(id,name))),next"
for year, playlist_id in annual_playlists.items():
	if not playlist_id: continue

	print(f"Processando músicas do ano {year}")

	results = sp_client.playlist_items(playlist_id, limit=100, fields=FIELDS, additional_types=["track"])

	all_items = results["items"]
	while results.get("next"):
		results = sp_client.next(results)
		all_items.extend(results["items"])

	for item in all_items:
		track = item.get("item") or item.get("track")
		if not track or not track.get("id"): continue

		artists = track.get("artists") or []
		artist = artists[0] if artists else None

		release_date_raw = track["album"]["release_date"]
		release_year = int(release_date_raw.split("-")[0]) if release_date_raw else None

		track_id_counter += 1
		music_data.append({
			"track_id": f"track_{track_id_counter:04d}",
			"artist_id": artist.get("id") if artist else None,
			"artist_name": artist.get("name") if artist else "Unknown",
			"track_name": track["name"],
			"duration_ms": track["duration_ms"],
			"release_year": release_year,
			"top_year": year,
			"explicit": track.get("explicit"),
			"album_type": track["album"].get("album_type"),
			"artist_count": len(artists)
		})

dataframe_spotify = pandas.DataFrame(music_data).astype({"release_year": "Int64"})

output_path = get_dataset_path()
dataframe_spotify.to_csv(output_path, index=False)
print(f"DataSet de 2010 a 2025 gerado com sucesso em {output_path}!")
print(f"Total de músicas: {len(dataframe_spotify)}")
print(dataframe_spotify["album_type"].value_counts())