import io
import re
import unicodedata
import difflib
import requests
import pandas as pandas

from data_preparation import get_output_dir, DATASET_FILE

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
HEADERS = {"User-Agent": "Mozilla/5.0 (trabalho academico de estatistica)"}
SIMILARITY_THRESHOLD = 0.85

# Equivalências conferidas manualmente: título normalizado no Spotify -> título normalizado oficial
MANUAL_EQUIVALENCES = {
	"fkin perfect": "fuckin perfect", # título censurado no Spotify
	"nias in paris": "niggas in paris", # título censurado no Spotify
	"fkin problems": "fuckin problems", # título censurado no Spotify
	"hot ngga": "hot boy", # título censurado pela Billboard
	"my nigga": "my hitta", # título censurado pela Billboard
	"py fairy": "pussy fairy", # título censurado no Spotify
	"plain jane remix": "plain jane", # remix creditado pela Billboard (feat. Nicki Minaj)
	"savage remix": "savage", # remix creditado pela Billboard (feat. Beyoncé)
	"beat box 2": "beat box 2 beat box 3", # entrada única da Billboard para as duas versões
	"hrs hrs": "hrs and hrs", # "&" grafado como "and"
	"shakira bzrp music sessions vol 5366": "bzrp music sessions vol 53", # título completo no Spotify
}

DIVERGENCE_TYPES = ["no CSV, não encontrada na lista oficial", "na lista oficial, ausente do CSV",
                    "correspondência aproximada (conferir)"]

def normalize_title(title):
	title = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode()
	title = title.lower()
	title = re.sub(r"[\(\[].*?[\)\]]", "", title)
	title = re.sub(r"\s+-\s+.*$", "", title)
	title = re.sub(r"\b(feat|ft|featuring|with)\b.*$", "", title)
	title = re.sub(r"[^a-z0-9 ]", "", title)
	return re.sub(r"\s+", " ", title).strip()

def fetch_official_list(year):
	response = requests.get(WIKIPEDIA_URL.format(year=year), headers=HEADERS, timeout=30)
	response.raise_for_status()
	for table in pandas.read_html(io.StringIO(response.text)):
		title_columns = [c for c in table.columns if "title" in str(c).lower()]
		if title_columns and len(table) >= 100:
			artist_columns = [c for c in table.columns if "artist" in str(c).lower()]
			return pandas.DataFrame({
				"official_title": table[title_columns[0]].astype(str).str.strip('"'),
				"official_artist": table[artist_columns[0]].astype(str) if artist_columns else "",
			})
	raise ValueError(f"Tabela não encontrada para {year}")

def best_match(title, candidates):
	scores = [difflib.SequenceMatcher(None, title, c).ratio() for c in candidates]
	best_position = max(range(len(scores)), key=scores.__getitem__)
	return best_position, scores[best_position]

def describe(song=None, official_row=None):
	return (f"{song['track_name']} — {song['artist_name']}" if song is not None else "",
	        f"{official_row['official_title']} — {official_row['official_artist']}" if official_row is not None else "")

dataset_dir = get_output_dir("dataset")
songs = pandas.read_csv(f"{dataset_dir}/{DATASET_FILE}")
report = []

for year in sorted(songs["top_year"].unique()):
	official = fetch_official_list(year)
	official["normalized"] = official["official_title"].map(normalize_title)
	year_songs = songs[songs["top_year"] == year].copy()
	year_songs["normalized"] = year_songs["track_name"].map(normalize_title)

	matched_official = set()
	counts = {"exata": 0, "aproximada": 0, "manual": 0}

	for _, song in year_songs.iterrows():
		available = [i for i in official.index if i not in matched_official]
		equivalent = MANUAL_EQUIVALENCES.get(song["normalized"])

		if equivalent is not None:
			candidates = [i for i in available if official.loc[i, "normalized"] == equivalent]
			if candidates:
				matched_official.add(candidates[0])
				counts["manual"] += 1
				spotify, oficial = describe(song, official.loc[candidates[0]])
				report.append({"ano": year, "tipo": "equivalência conferida manualmente",
				               "spotify": spotify, "oficial": oficial, "similaridade": None})
				continue

		position, score = best_match(song["normalized"], [official.loc[i, "normalized"] for i in available])
		index = available[position]
		if score >= SIMILARITY_THRESHOLD:
			matched_official.add(index)
			if score == 1:
				counts["exata"] += 1
			else:
				counts["aproximada"] += 1
				spotify, oficial = describe(song, official.loc[index])
				report.append({"ano": year, "tipo": "correspondência aproximada (conferir)",
				               "spotify": spotify, "oficial": oficial, "similaridade": round(score, 2)})
		else:
			spotify, _ = describe(song)
			report.append({"ano": year, "tipo": "no CSV, não encontrada na lista oficial",
			               "spotify": spotify, "oficial": "", "similaridade": round(score, 2)})

	for index in official.index:
		if index not in matched_official:
			_, oficial = describe(official_row=official.loc[index])
			report.append({"ano": year, "tipo": "na lista oficial, ausente do CSV",
			               "spotify": "", "oficial": oficial, "similaridade": None})

	print(f"{year}: {len(year_songs)} músicas | {counts['exata']} exatas | "
	      f"{counts['aproximada']} aproximadas | {counts['manual']} por equivalência manual")

report_dataframe = pandas.DataFrame(report, columns=["ano", "tipo", "spotify", "oficial", "similaridade"])
report_path = f"{dataset_dir}/billboard_validation_report.csv"
report_dataframe.to_csv(report_path, index=False, encoding="utf-8-sig")

unexplained = report_dataframe["tipo"].isin(DIVERGENCE_TYPES).sum()
print(f"\nRegistros no relatório: {len(report_dataframe)}")

if len(report_dataframe):
	print(report_dataframe["tipo"].value_counts())

print(f"Divergências não explicadas: {unexplained}")
print(f"Relatório salvo em: {report_path}")