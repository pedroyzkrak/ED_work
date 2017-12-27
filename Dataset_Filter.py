import pandas as pd

tracks = pd.read_csv('tracks.csv', low_memory=False, skiprows = [1])
tracks.rename(columns={"Unnamed: 0" : "track_id"}, inplace=True)
features = pd.read_csv('features.csv', low_memory=False, skiprows=[1,2,3])
pd.set_option('max_columns',20)
subset_tracks = tracks.index[(tracks["set.1"] == 'medium') | (tracks["set.1"] == 'small')]
tracks = tracks.loc[subset_tracks]
tracks = tracks[["track_id","track","track.7","track.19","track.5"]] #track_id, bitrate, genre_top, title, duration

genres = pd.read_csv('genres.csv', low_memory=False)
genres = genres[["genre_id", "parent", "title"]]

columns_dict = {}
for column in features.columns:
    if '.' in column and column.split('.')[0] in columns_dict.keys():
        columns_dict[column.split('.')[0]].append(column)
    else:
        columns_dict[column] = [column]

required_columns = columns_dict['mfcc'] + columns_dict['spectral_contrast'] \
                   + columns_dict['chroma_cens'] + columns_dict['spectral_centroid'] \
                   + columns_dict['zcr'] + columns_dict['tonnetz']

features = features[required_columns]

tracks_and_features = pd.concat([tracks,features], axis=1).dropna()

tracks_and_features.to_csv("Filtered_Tracks.csv", index=False, header=False)
genres.to_csv("Filtered_Genres.csv", index=False, header=False)
