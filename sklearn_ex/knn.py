import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import base64
import io
from matplotlib.pyplot import imread
import codecs
from IPython.display import HTML

movies = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits = pd.read_csv('../input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies.head()
movies.describe()
credits.head()
credits.describe()

# changing the genres column from json to string
movies['genres'] = movies['genres'].apply(json.loads)
for index, i in zip(movies.index, movies['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))  # the key 'name' contains the name of the genre
    movies.loc[index, 'genres'] = str(list1)

# changing the keywords column from json to string
movies['keywords'] = movies['keywords'].apply(json.loads)
for index, i in zip(movies.index, movies['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index, 'keywords'] = str(list1)

# changing the production_companies column from json to string
movies['production_companies'] = movies['production_companies'].apply(json.loads)
for index, i in zip(movies.index, movies['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index, 'production_companies'] = str(list1)

# changing the cast column from json to string
credits['cast'] = credits['cast'].apply(json.loads)
for index, i in zip(credits.index, credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index, 'cast'] = str(list1)

# changing the crew column from json to string
credits['crew'] = credits['crew'].apply(json.loads)


def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']


credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew': 'director'}, inplace=True)

movies.head()

movies.iloc[25]