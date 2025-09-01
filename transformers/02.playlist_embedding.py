import os
import pandas as pd

print(os.getcwd())


f = open("./data/yes_playlist/yes_complete/train.txt")
train_data = f.readlines()
f.close()

playlists = [ s.rstrip().split() for s in train_data if len(s.strip()) > 0]
print( len(playlists) )

f = open("./data/yes_playlist/yes_complete/song_hash.txt")
song_data = f.readlines()
f.close()

songs = [s.rstrip().split("\t") for s in song_data]
songs_df = pd.DataFrame(songs, columns=["id", "title", "artist"])
songs_df = songs_df.set_index("id")

print( songs_df.head() )


from gensim.models import Word2Vec

# train Word2Vec model
model = Word2Vec(sentences=playlists, vector_size=32, window=20, negative=50, min_count=1, workers=4)

song_id = 2172

#output = model.wv.most_similar(positive=str(song_id), topn=10)


import numpy as np

def print_recommendations(song_id):
    similar_songs = np.array (model.wv.most_similar(positive=str(song_id), topn=5))[:,0]
    return songs_df.iloc[similar_songs]

print( print_recommendations(2172) )