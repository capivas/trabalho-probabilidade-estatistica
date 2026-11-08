# A era do consumo rápido

Análise probabilística da duração e do conteúdo explícito das músicas de maior sucesso nos Estados Unidos (2010–2025).

Trabalho de Complementação da disciplina de Probabilidade e Estatística.

## Questão de investigação

As músicas de maior sucesso tornaram-se mais curtas e mais frequentemente explícitas na era do consumo rápido (2020–2025), em comparação com a década de 2010, de forma compatível com as mudanças trazidas pelo streaming e pela descoberta musical em vídeos curtos?

**Hipóteses de pesquisa**

1. As músicas de maior sucesso da era recente são, em média, mais curtas que as da década de 2010.
2. A proporção de músicas com conteúdo explícito é maior na era recente do que na década de 2010.

## Principais resultados

| Medida | 2010–2019 | 2020–2025 |
|---|---|---|
| Músicas com menos de 3 minutos | 7,9% | 33,3% |
| Duração média | 3,71 min | 3,30 min |
| Músicas com conteúdo explícito | 33,7% | 45,0% |

- A duração média caiu 24,4 segundos (IC 95%: 20,4 a 28,5 s), com associação moderada entre a era e a duração inferior a 3 minutos (φ = 0,32). O resultado se mantém sem as músicas natalinas.
- A mudança na duração começou em 2018–2019, antes do recorte adotado.
- O aumento do conteúdo explícito é significativo, mas a associação é fraca (φ = 0,11).
- Um classificador Naive Bayes com essas duas características acertou a era de 70,1% das músicas em validação cruzada, contra 62,5% de um classificador de referência.

O relatório completo está em [`relatorio/`](relatorio/).

## Dados

- **Fonte:** listas anuais Billboard Year-End Hot 100 de 2010 a 2025 (16 listas, 1600 músicas).
- **Coleta:** metadados obtidos pela Web API do Spotify, a partir de playlists que reproduzem cada lista anual.
- **Validação:** as listas coletadas foram comparadas com as edições oficiais reproduzidas na Wikipédia. Todas as 1600 músicas foram associadas a uma entrada oficial: 1589 por correspondência exata de título e 11 por equivalências conferidas manualmente (títulos censurados, grafias distintas e versões creditadas pela Billboard). As equivalências estão documentadas em `data_verification.py` e no relatório `dataset/billboard_validation_report.csv`.

O conjunto de dados final está em `dataset/top100_2010s_2020s.csv`.

| Coluna | Descrição |
|---|---|
| `track_id` | Código sequencial da música |
| `artist_id` | Código do artista principal no Spotify |
| `artist_name` | Nome do artista principal |
| `track_name` | Nome da música |
| `duration_ms` | Duração, em milissegundos |
| `release_year` | Ano de lançamento do álbum ou single |
| `top_year` | Ano da lista em que a música aparece |
| `explicit` | Se a versão da faixa contém conteúdo explícito |
| `album_type` | Tipo de lançamento da versão (álbum, single ou coletânea) |
| `artist_count` | Número de artistas creditados |

As colunas derivadas (`duration_min`, `era` e `is_short`) são criadas em `data_preparation.py`.

## Estrutura do repositório

```
.
├── dataset/
│   ├── top100_2010s_2020s.csv           # conjunto de dados final
│   └── billboard_validation_report.csv  # equivalências da validação
├── figures/                             # gráficos gerados pelas análises
├── relatorio/                           # relatório em PDF
├── data_preparation.py                  # leitura dos dados e variáveis derivadas (módulo comum)
├── 01_spotify_data_extraction.py        # coleta pela Web API do Spotify
├── data_verification.py                 # validação com as listas oficiais
├── 02_probabilities.py                  # probabilidades clássicas e condicionais
├── 03_distributions.py                  # distribuições binomial e normal
├── 04_bayes.py                          # estatística bayesiana e Naive Bayes
├── 05_complementary_tests.py            # testes complementares e análise de sensibilidade
├── requirements.txt
└── .env.example
```

## Como executar

### 1. Preparar o ambiente

Requer Python 3.11 ou superior.

```bash
python -m venv venv
```

Ative o ambiente virtual:

- Windows (cmd): `venv\Scripts\activate`
- Windows (PowerShell): `venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 2. Reproduzir as análises

As análises usam o conjunto de dados já incluído no repositório e **não exigem acesso à API do Spotify**. Execute os scripts na ordem:

```bash
python 02_probabilities.py
python 03_distributions.py
python 04_bayes.py
python 05_complementary_tests.py
```

Os resultados são exibidos no terminal, e os gráficos são salvos na pasta `figures/`. A validação cruzada usa semente fixa, então os resultados são idênticos aos do relatório.

### 3. Refazer a coleta (opcional)

Refazer a coleta sobrescreve o arquivo `dataset/top100_2010s_2020s.csv`. Pelas regras atuais da Web API do Spotify, ela exige:

- conta Spotify **Premium** do responsável pela aplicação;
- uma aplicação criada no [Spotify for Developers](https://developer.spotify.com/dashboard), com a opção **Web API** e a Redirect URI `http://127.0.0.1:8888/callback`;
- playlists **da própria conta**, já que a API só retorna o conteúdo de playlists do usuário autenticado.

Passos:

1. Copie `.env.example` para `.env` e preencha com o Client ID e o Client Secret da sua aplicação.
2. Copie para a sua conta as playlists da tabela abaixo e substitua os IDs no dicionário `annual_playlists` de `01_spotify_data_extraction.py`.
3. Execute `python 01_spotify_data_extraction.py`. Na primeira execução, o navegador abrirá para autorizar o acesso.
4. Execute `python data_verification.py` para validar as listas. O resultado esperado é "Divergências não explicadas: 0".

| Ano | Playlist | Ano | Playlist |
|---|---|---|---|
| 2010 | [link](https://open.spotify.com/playlist/5ZS9pYgTmybr3ci7LdZVRW) | 2018 | [link](https://open.spotify.com/playlist/3NogZcZzzFF3Bgmi2KMDvT) |
| 2011 | [link](https://open.spotify.com/playlist/2eRQ2GefRVucxxdPfFw92w) | 2019 | [link](https://open.spotify.com/playlist/6cCdvEVpN9RKWTgwz3UjhH) |
| 2012 | [link](https://open.spotify.com/playlist/5fkMnPqY0ilax2ZEAyhvQl) | 2020 | [link](https://open.spotify.com/playlist/6ho6OCypoafwr3HEqbQXpE) |
| 2013 | [link](https://open.spotify.com/playlist/7kPgVGmUNL8ZPLLXjyxsYJ) | 2021 | [link](https://open.spotify.com/playlist/25vaJELb9hxFqSsZn0YgiN) |
| 2014 | [link](https://open.spotify.com/playlist/4R0umj6e3u1JY6XD4gZZnt) | 2022 | [link](https://open.spotify.com/playlist/3G3wk1cUcvhQcQkoAuQawZ) |
| 2015 | [link](https://open.spotify.com/playlist/01DHv1EhNLmLZOZ7Xft2kd) | 2023 | [link](https://open.spotify.com/playlist/7E0xrKqCNtVUrKlFN9OmRh) |
| 2016 | [link](https://open.spotify.com/playlist/7yL74sMdtghrje0PngyyeB) | 2024 | [link](https://open.spotify.com/playlist/1YnHrEdgKANonoC6xejEPo) |
| 2017 | [link](https://open.spotify.com/playlist/2C2mTl6wW8cUnwEl29whXr) | 2025 | [link](https://open.spotify.com/playlist/25dBkFPHERTFYP0e7ue0iL) |

Como a validação compara apenas os títulos, mudanças futuras no catálogo do Spotify (como a troca de versões de uma faixa) podem alterar levemente os valores coletados.

## Ferramentas

| Ferramenta | Versão | Uso |
|---|---|---|
| Python | 3.11.4 | Linguagem |
| pandas | 3.1.0 | Manipulação dos dados |
| NumPy | 2.4.6 | Operações numéricas |
| SciPy | 1.17.1 | Testes estatísticos e distribuições |
| Matplotlib | 3.11.2 | Gráficos |
| Spotipy | 2.26.0 | Acesso à Web API do Spotify |
| requests | 2.34.2 | Download das listas oficiais |
| lxml | 6.1.3 | Leitura das tabelas das listas oficiais |
| python-dotenv | 1.2.4 | Leitura das credenciais do arquivo `.env` |

## Autor

Victor Hugo Ferreira Camargo — Análise e Desenvolvimento de Sistemas, UniEVANGÉLICA - Universidade Evangélica de Goiás, 2026.
