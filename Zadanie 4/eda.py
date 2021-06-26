from matplotlib import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def english_language(df):
    english_dict = dict(df.english.value_counts())
    labels = 'Áno', 'Nie'
    sizes = [english_dict[1], english_dict[0]]

    plt.pie(sizes, explode=(0, 0.2), labels=labels, autopct='%1.1f%%', startangle=140)

    plt.axis('equal')
    plt.title('Podpora anglického jazyka')
    plt.show()


def age(df):
    age_dict = dict(df.required_age.value_counts())
    labels = 'Nie', 'Áno'
    sizes = [age_dict[0], age_dict[18] + age_dict[16] + age_dict[12] + age_dict[7] + age_dict[3]]

    plt.pie(sizes, explode=(0, 0.2), labels=labels, autopct='%1.1f%%', startangle=140)

    plt.axis('equal')
    plt.title('Vekové obmedzenie')
    plt.show()


def age_restriction(df):
    age_dict = dict(df.required_age.value_counts())
    labels = '18', '16', '12', '7', '3'
    sizes = [age_dict[18], age_dict[16], age_dict[12], age_dict[7], age_dict[3]]

    plt.pie(sizes, explode=(0, 0.05, 0.05, 0.1, 0.2), labels=labels, autopct='%1.1f%%', startangle=140)

    plt.axis('equal')
    plt.title('Vekove rozdelenie')
    plt.show()


def owners(df):
    graph = sns.countplot(data=df, y='owners', orient='h', dodge=False,  hue='owners',
                          order=df.owners.value_counts().index)
    h, l = graph.get_legend_handles_labels()
    labels = ["20k alebo menej", "20k-50k", "50k-100k", "100k-200k", "200k-500k", "500k-1mil", "1mil-2mil",
              "2mil-5mil", "5mil-10mil", "10mil-20mil", "20mil-50mil", "50mil-100mil", "100mil-200mil"]
    graph.legend(h, labels, title="Majitelia", loc="lower right")
    graph.set_xscale('log')
    graph.set_title("Predaje hier")
    plt.show()


def release_dates(df):
    plt.figure(figsize=(12, 6))
    years = df.groupby(df.release_date.dt.year.rename('release_year')) \
        .agg('count').release_date.rename('count')
    until_2019 = years[years.index < 2019]

    graph = sns.barplot(y=until_2019, x=until_2019.index)
    graph.set_title("Množstvo vydaných hier od času")
    plt.xlabel('Rok')
    plt.ylabel('Počet hier')
    plt.show()


def platform(df):
    linux_count = df.linux.value_counts()[1]
    win_count = df.windows.value_counts()[1]
    mac_count = df.mac.value_counts()[1]

    data = np.array([win_count, mac_count, linux_count])
    platform_series = pd.Series(data, index=['windows', 'mac', 'linux'])

    graph = sns.barplot(y=platform_series.index, x=platform_series / len(df))
    graph.set_title("Podpora hier pre jednotlivé OS")

    values = graph.get_xticks()
    graph.set_xticklabels(['{:,.0%}'.format(x) for x in values])

    plt.ylabel('OS')
    plt.show()


def genres(df):
    plt.figure(figsize=(12, 8))

    genres = ["Action", "Free to Play", "Strategy", "Adventure", "Indie", "RPG", "Casual", "Simulation", "Racing",
              "Animation", "Sports", "Education"]

    genres2 = ["Indie", "Action", "Casual", "Adventure", "Strategy", "Simulation", "RPG", "Free to Play", "Sports",
               "Racing", "Animation", "Education"]

    data = []
    for genre in genres:
        data.append(df[genre].value_counts()[1])

    data.sort(reverse=True)
    genres_series = pd.Series(data, index=genres2)

    graph = sns.barplot(y=genres_series.index, x=genres_series / len(df))

    values = graph.get_xticks()
    graph.set_xticklabels(['{:,.0%}'.format(x) for x in values])
    graph.set_title("Zastúpenie hier podľa žánrov")

    for inx, p in enumerate(graph.patches):
        graph.annotate(data[inx], (p.get_x() + p.get_width(), p.get_y() + 0.7),
                    xytext=(5, 10), textcoords='offset points')

    plt.ylabel('Žánre')

    plt.show()


def top_played_games(df):
    plt.figure(figsize=(15, 8))

    game_names = []
    playtimes = []

    for row in df.itertuples():
        game_names.append(row.name)
        playtimes.append(row.average_playtime)

    genres_series = pd.Series(playtimes, index=game_names)
    genres_series = genres_series.sort_values(ascending=False)
    genres_series = genres_series[:30]
    graph = sns.barplot(y=genres_series.index, x=genres_series)

    graph.set_title("Top 30 najhranejších hier")

    for inx, p in enumerate(graph.patches):
        graph.annotate(genres_series[inx], (p.get_x() + p.get_width()+1, p.get_y() + 0.9),
                       xytext=(5, 10), textcoords='offset points')

    plt.ylabel('Hry')
    plt.show()


def developer(df):
    plt.figure(figsize=(15, 8))
    plt.title(f'Top 30 developérov s najväčším množstvom hier')
    sns.countplot(y="developer", data=df,
                  order=df.developer.value_counts().iloc[:30].index)
    plt.show()


def publisher(df):
    plt.figure(figsize=(15, 8))
    plt.title(f'Top 30 vydavateľov s najväčším možstvom hier')
    sns.countplot(y="publisher", data=df,
                  order=df.publisher.value_counts().iloc[:30].index)
    plt.show()


def price(df):
    games = dict(df['Free to Play'].value_counts())
    labels = 'Zadarmo', 'Platené'
    sizes = [games[1], games[0]]

    plt.pie(sizes, explode=(0, 0.5,), labels=labels, autopct='%1.1f%%', startangle=140)

    plt.axis('equal')
    plt.title('Free to Play vs Platené hry')
    plt.show()
