from timeit import default_timer
import requests
import pandas as pd
import re
import os

os.system('cls') if os.name =='nt' else os.system('clear')

with open('data/imdb/top250.txt') as f:
    imdb_movies_load = f.read().split('\n')

topics = imdb_movies_load[0:5]
wiki_data_movies = {
    'X': [],
    'y': [],
    }

words_sum = 0
position = 0
for topic in topics:
    try:
        wikipedia_request = requests.get(
            f'https://pt.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&'+
            f'titles={topic}').json()

        data = wikipedia_request['query']['pages'][[*wikipedia_request['query']['pages']][0]]
        
        summary = re.sub(r'[^\w]', ' ', data['extract'])
        tittle = data['title']

        words_sum += len(summary.split())

        wiki_data_movies['X'].append(summary)
        wiki_data_movies['y'].append(tittle)
        
        position += 1

        print(f'{tittle} ({len(summary.split())} palavras) - {summary[0:50]}')
    except:
        print(f' - {topic} X X X X X X X X X X X X X X X X X X X X X X X')

print(words_sum/len(topics))
pd.DataFrame(wiki_data_movies).to_csv('data/wikipedia/wiki_imdb_summaries.csv', index=False)