import pandas as pd
import plotly as plotly
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
from sklearn.decomposition import PCA
from datetime import date
from sklearn.cluster import DBSCAN
import eda
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.mode.chained_assignment = None


def read_csv(name):
    return pd.read_csv(name, encoding="utf-8")


def merge_data(df1, df2):
    return pd.merge(df1, df2, on=['appid'])


# String na DateTime
def str_to_datetime(df):
    df.release_date = df.release_date.map(lambda s: dt.datetime.strptime(s, "%Y-%m-%d"))
    return df


# Tags processing
def interesting_tags(df, tags_df):
    tags = ["2d", "2.5d", "3d", "first_person", "third_person", "vr", "e_sports"]

    for tag in tags:
        df[tag] = tags_df.apply(lambda row: contains_tag(row, tag), axis=1)
    return df


def contains_tag(row, tag):
    if row[tag] > 0:
        return 1
    else:
        return 0


# Developer processing
def code_developer(df):
    # df.drop(['publisher'], axis=1, inplace=True)
    counts_no_games = df['developer'].value_counts()
    ratings_sum = {}

    df['number_of_games'] = df.apply(lambda row: count_developer_games(counts_no_games, row), axis=1)
    df['number_of_ratings_developer'] = df.apply(lambda row: process_developer_rating(ratings_sum, df, row), axis=1)
    return df


def count_developer_games(counts, row):
    return counts[row['developer']]


def process_developer_rating(dictionary, df, row):

    if row['developer'] not in dictionary.keys():
        dictionary[row['developer']] = sum_developer_rating(df, row)

    return dictionary[row['developer']]


def sum_developer_rating(df, row):
    count_positive_ratings = df.loc[df['developer'] == row['developer'], 'positive_ratings'].sum()
    count_negative_ratings = df.loc[df['developer'] == row['developer'], 'negative_ratings'].sum()
    return count_positive_ratings + count_negative_ratings


# Owners processing
def owners_to_int(df):
    return df.replace({'owners': {'0-20000': 0, '20000-50000': 1, '50000-100000': 2, '100000-200000': 3,
                                  '200000-500000': 4, '500000-1000000': 5, '1000000-2000000': 6, '2000000-5000000': 7,
                                  '5000000-10000000': 8, '10000000-20000000': 9, '20000000-50000000': 10,
                                  '50000000-100000000': 11, '100000000-200000000': 12}})


# Platforms processing
def split_platform(df):
    df['windows'] = df.apply(lambda row: supports_platform(row, 'windows'), axis=1)
    df['mac'] = df.apply(lambda row: supports_platform(row, 'mac'), axis=1)
    df['linux'] = df.apply(lambda row: supports_platform(row, 'linux'), axis=1)
    df.drop(['platforms'], axis=1, inplace=True)
    return df


def supports_platform(row, platform):
    if platform in row['platforms']:
        return 1
    else:
        return 0


# Genres processing
def code_genres(df):
    genres = ["Action", "Free to Play", "Strategy", "Adventure", "Indie", "RPG", "Casual", "Simulation", "Racing",
              "Animation", "Sports", "Education"]

    df = df[(df['genres'].str.contains("Action|Free to Play|Strategy|Adventure|Indie|RPG|Casual" +
            "|Simulation|Racing|Animation|Sports|Education"))]

    for genre in genres:
        df[genre] = df.apply(lambda row: contains_genre(row, genre), axis=1)

    df.drop(['genres'], axis=1, inplace=True)
    return df


def contains_genre(row, genre):
    if genre in row['genres']:
        return 1
    else:
        return 0


# Categories processing
def split_categories(df):
    df = df[(df['categories'].str.contains("(?i)mmo|co-op|single-player|multi-player"))]
    df.loc[df['categories'].str.contains("(?i)mmo|co-op"), 'categories'] = 'multi-player'

    df['Single-player'] = df.apply(lambda row: what_category(row, 'single-player'), axis=1)
    df['Multi-player'] = df.apply(lambda row: what_category(row, 'multi-player'), axis=1)

    df.drop(['categories'], axis=1, inplace=True)
    return df


def what_category(row, category):
    if category in row['categories'].lower():
        return 1
    else:
        return 0


def find_all_unique_genres(df):
    uniqueGenresColumns = df.genres.unique()
    uniqueGenres = set()
    for column in uniqueGenresColumns:
        for genre in column.split(";"):
            uniqueGenres.add(genre)


def data_processing(df, df_tag):
    # data = merge_data(df, df_tag)
    data = df.set_index(['appid'])
    data = str_to_datetime(data)
    data = interesting_tags(data, df_tag)
    data = code_developer(data)
    data.drop(['achievements'], axis=1, inplace=True)
    data = owners_to_int(data)
    data = split_platform(data)
    data = code_genres(data)
    data = split_categories(data)
    data.to_csv("steam_data_encoded.csv", index=True)
    return data


def remove_strig_vals(df):
    df.drop(['name'], axis=1, inplace=True)
    df.drop(['developer'], axis=1, inplace=True)
    df.drop(['publisher'], axis=1, inplace=True)
    df.drop(['release_date'], axis=1, inplace=True)


def scale(df):
    remove_strig_vals(df)
    slaced_data = StandardScaler().fit_transform(df)
    return pd.DataFrame(slaced_data, columns=df.columns, index=df.index)


# EDA
def create_eda(df):
    df = str_to_datetime(df)
    eda.english_language(df)
    eda.age(df)
    eda.age_restriction(df)
    eda.owners(df)
    eda.release_dates(df)
    eda.platform(df)
    eda.genres(df)
    eda.top_played_games(df)
    eda.developer(df)
    eda.publisher(df)
    eda.price(df)


def mini_batch_KMeans(df, dfAll):
    number_of_cluters = 7
    km = MiniBatchKMeans(number_of_cluters, random_state=0, batch_size=50, init='k-means++', max_iter=300, verbose=True)
    km_model = km.fit(df)
    kmeanlabels = km.labels_

    # df['cluster_id'] = kmeanlabels
    dfAll['cluster_id'] = kmeanlabels

    print(dfAll.cluster_id.value_counts())

    for name in dfAll.columns:
        if name in "cluster_id":
            continue
        x = dfAll.groupby('cluster_id').mean()
        ax = sns.barplot(x=x.index, y=name, data=x)
        plt.show()

    return kmeanlabels


def db_scan(df, dfAll):

    dbs = DBSCAN(eps=3, min_samples=50)
    dbs_model = dbs.fit(df)
    kmeanlabels = dbs.labels_

    # df['cluster_id'] = kmeanlabels
    dfAll['cluster_id'] = kmeanlabels

    print(dfAll.cluster_id.value_counts())

    for name in dfAll.columns:
        if name in "cluster_id":
            continue
        x = dfAll.groupby('cluster_id').mean()
        ax = sns.barplot(x=x.index, y=name, data=x)
        plt.show()

    return kmeanlabels


def pca(dfs, cluster, name, filename):
    pca = PCA(n_components=3, whiten=True)
    components = pca.fit_transform(dfs)

    components = pd.DataFrame(data=components)
    components['cluster'] = cluster
    components['name'] = name.reset_index(drop=True)
    total_var = pca.explained_variance_ratio_.sum() * 100

    fig = px.scatter_3d(
        components, x=0, y=1, z=2, hover_data=name, color=components['cluster'],
        title=f'Total Explained Variance: {total_var:.2f}%',

    )

    plotly.offline.plot(fig, filename=filename)


def heat_man(df):
    plt.figure(figsize=(16, 15))

    ax = sns.heatmap(
        df.corr(),
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(200, 300, n=300),
        square=True,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='right'
    )

    plt.show()


if __name__ == '__main__':
    steam = read_csv('steam.csv')
    steam_tag = read_csv('steamspy_tag_data.csv')
    # merged_steam_data = merge_data(steam, steam_tag)
    # processed_data = str_to_datetime(processed_data)

    # Treba vybrat jednu z moznosti 1/2

    # 1.Spracovanie vstupnych suborov
    # processed_data = data_processing(steam, steam_tag)

    # 2.CSV uz s uz upravenymi datami aby nebolo nutne pri kazdom spusteni spracovavat data
    steam_tag = steam_tag.set_index(['appid'])
    processed_data = read_csv('steam_data_encoded.csv')

    #EDA
    #create_eda(processed_data)

    # Clustering
    processed_data = processed_data.set_index(['appid'])

    # heat_man(processed_data)
    name = processed_data[['name']]
    scaled_processed_data = scale(processed_data)
    scaled_processed_data_copy = scaled_processed_data.copy()
    cluster_kmeans = mini_batch_KMeans(scaled_processed_data, processed_data)
    cluster_dbscan = db_scan(scaled_processed_data, processed_data)
    pd.DataFrame(cluster_kmeans).to_csv('cluster_kmeans.csv', index=None)
    pd.DataFrame(cluster_dbscan).to_csv('cluster_dbscan.csv', index=None)
	
    pca(scaled_processed_data_copy, read_csv('cluster_kmeans.csv').to_numpy(), name, 'kmeans_plot.html')
    pca(scaled_processed_data_copy, read_csv('cluster_dbscan.csv').to_numpy(), name, 'dbscan_plot.html')

    print('DONE')
