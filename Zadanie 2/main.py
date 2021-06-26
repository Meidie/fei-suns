import pandas as pd
import numpy as np
import plotly as plotly
from sklearn import preprocessing
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix


def read_csv(name):
    return pd.read_csv(name, index_col=0, encoding="utf-8")


def replace_string_values(df):
    df = df.replace("edm", 0)
    df = df.replace("latin", 1)
    df = df.replace("pop", 2)
    df = df.replace("r&b", 3)
    df = df.replace("rap", 4)
    df = df.replace("rock", 5)

    return df


def clear_wrong_data(df):

    # df = df[df['playlist_genre'] == 'rock']

    # EDM
    df = df.drop(df[df['playlist_name'] == '3 am tears  '].index)
    df = df.drop(df[df['playlist_name'] == 'BEST OF 2016 - TOP HITS | NEW MUSIC - POP | EDM | HIP HOP | R&B'].index)
    df = df.drop(df[df['playlist_name'] == 'DJ Mix - Frat Party (Rap, Pop, EDM)'].index)
    df = df.drop(df[df['playlist_name'] == 'Pop Hits 2000-2020'].index)

    df = df.drop(df[(df['playlist_genre'] == 'edm') & (df['energy'] < 0.2)].index)
    df = df.drop(df[(df['playlist_genre'] == 'edm') & (df['loudness'] < -18.5)].index)
    df = df.drop(df[(df['playlist_genre'] == 'edm') & (df['speechiness'] > 0.52)].index)
    df = df.drop(df[(df['playlist_genre'] == 'edm') & (df['tempo'] > 152)].index)

    # Latin
    df = df.drop(df[df['playlist_name'] == 'Tropical Nights'].index)
    df = df.drop(df[df['playlist_name'] == 'Latin Pop Rising'].index)
    df = df.drop(df[df['playlist_name'] == 'Cachacas tropicales'].index)

    df = df.drop(df[(df['playlist_genre'] == 'latin') & (df['danceability'] < 0.280)].index)
    df = df.drop(df[(df['playlist_genre'] == 'latin') & (df['energy'] < 0.195)].index)
    df = df.drop(df[(df['playlist_genre'] == 'latin') & (df['loudness'] < -19)].index)
    df = df.drop(df[(df['playlist_genre'] == 'latin') & (df['liveness'] > 0.57)].index)

    # Pop
    df = df.drop(df[df['playlist_name'] == 'Indie Pop Nation'].index)

    df = df.drop(df[(df['playlist_genre'] == 'pop') & (df['loudness'] < -20)].index)
    df = df.drop(df[(df['playlist_genre'] == 'pop') & (df['speechiness'] > 0.49)].index)

    # R&B

    # Rap
    df = df.drop(df[df['playlist_name'] == 'Badass Rock'].index)
    df = df.drop(df[df['playlist_name'] == 'TRAPPED IN A HORROR MOVIE AND IDK IF IM THE VICTIM OR THE KILLER '].index)

    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['danceability'] < 0.286)].index)
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['energy'] < 0.2)].index)
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['loudness'] < -19)].index)
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['speechiness'] > 0.6)].index)
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['liveness'] > 0.628)].index)

    # Rock
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['loudness'] < -25)].index)
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['speechiness'] > 0.48)].index)
    df = df.drop(df[(df['playlist_genre'] == 'rap') & (df['liveness'] > 0.797)].index)

    '''
    fig = make_subplots(rows=3, cols=3, subplot_titles=("album:danceability", "album:energy", "album:loudness",
                                                        "album:speechiness", "album:acousticness",
                                                        "album:instrumentalness", "album:liveness", "album:valence",
                                                        "album:tempo"))
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['danceability'], mode="markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['energy'], mode="markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['loudness'], mode="markers"), row=1, col=3)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['speechiness'], mode="markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['acousticness'], mode="markers"), row=2, col=2)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['instrumentalness'], mode="markers"), row=2, col=3)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['liveness'], mode="markers"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['valence'], mode="markers"), row=3, col=2)
    fig.add_trace(go.Scatter(x=df['playlist_name'], y=df['tempo'], mode="markers"), row=3, col=3)
    '''
    #dff = df.apply(lambda x: True
    #if (x['playlist_genre'] == 'latin') else False, axis=1)
    #num_rows = len(dff[dff == True].index)
    #num_rows = len(df)
    #print('Number of Rows: ', num_rows)

    #plotly.offline.plot(fig, filename='plot.html')

    return df


def normalize(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    normalized_df.dropna()
    return normalized_df


def scale(df):
    names = df.columns
    scaled_df = preprocessing.StandardScaler().fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    scaled_df.dropna()
    return scaled_df


def split(df_X, df_Y ):
    X_train, X_valid, y_train, y_valid = train_test_split(df_X, df_Y, test_size=0.2)
    return X_train, X_valid, y_train, y_valid


def plot_result(training_data):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    fig.add_trace(go.Scatter(x=training_data.epoch, y=training_data.history['accuracy'],
                             name="Training accuracy", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=training_data.epoch, y=training_data.history['val_accuracy'],
                             name="Validation accuracy", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=training_data.epoch, y=training_data.history['loss'],
                             name="Training loss", mode="lines"), row=1, col=2)
    fig.add_trace(go.Scatter(x=training_data.epoch, y=training_data.history['val_loss'],
                             name="Validation loss", mode="lines"), row=1, col=2)

    plotly.offline.plot(fig, filename='result_plot.html')


if __name__ == '__main__':
    test_data = read_csv("test.csv")
    train_data = read_csv("train.csv")

    test_data = test_data.reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    train_data = replace_string_values(train_data)
    test_data = replace_string_values(test_data)

    cleared_train_data = clear_wrong_data(train_data)

    Xcols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
             'valence', 'tempo']

    train_X, valid_X, train_Y, valid_Y = split(train_data[Xcols], train_data[['playlist_genre']])

    test_X = test_data[Xcols]
    test_Y = test_data[['playlist_genre']]

    not_normalized_test_Y = test_Y
    not_normalized_train_Y = train_Y

    train_X = normalize(train_X)
    valid_X = normalize(valid_X)
    test_X = normalize(test_X)
    train_Y = normalize(train_Y)
    valid_Y = normalize(valid_Y)
    test_Y = normalize(test_Y)

    # Uloha 2

    model = keras.Sequential()
    model.add(layers.Dense(15, input_dim=9, activation='sigmoid'))
    model.add(layers.Dense(15, activation='sigmoid'))
    model.add(layers.Dense(6, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    y = pd.get_dummies(train_Y.playlist_genre)
    tst_Y = pd.get_dummies(test_Y.playlist_genre)
    val_Y = pd.get_dummies(valid_Y.playlist_genre)

    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    training = model.fit(train_X, y, epochs=300, batch_size=200, validation_data=(valid_X, val_Y), callbacks=[stop])
    pred_Y = model.predict(test_X)

    plot_result(training)
    print(metrics.classification_report(not_normalized_test_Y, np.argmax(pred_Y, axis=1)))


    # Uloha 3
    '''
    model = keras.Sequential()
    model.add(layers.Dense(200, input_dim=9, activation='sigmoid'))
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(layers.Dense(6, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    y = pd.get_dummies(train_Y.playlist_genre)
    tst_Y = pd.get_dummies(test_Y.playlist_genre)
    val_Y = pd.get_dummies(valid_Y.playlist_genre)

    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)

    training = model.fit(train_X, y, epochs=500, batch_size=200, validation_data=(valid_X, val_Y), callbacks=[stop])
    pred_Y = model.predict(test_X)

    plot_result(training)
    print(metrics.classification_report(not_normalized_test_Y, np.argmax(pred_Y, axis=1)))
    '''
    # Uloha 3 - L1/L2
    '''
    model = keras.Sequential()
    model.add(layers.Dense(200, input_dim=9, kernel_regularizer=keras.regularizers.l2(0.00001), activation='sigmoid'))
    model.add(layers.Dense(100, kernel_regularizer=keras.regularizers.l2(0.00001), activation='sigmoid'))
    model.add(layers.Dense(6, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    y = pd.get_dummies(train_Y.playlist_genre)
    tst_Y = pd.get_dummies(test_Y.playlist_genre)
    val_Y = pd.get_dummies(valid_Y.playlist_genre)

    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)

    training = model.fit(train_X, y, epochs=250, batch_size=200, validation_data=(valid_X, val_Y), callbacks=[stop])
    pred_Y = model.predict(test_X)

    plot_result(training)
    print(metrics.classification_report(not_normalized_test_Y, np.argmax(pred_Y, axis=1)))
    '''

    # Uloha 3 - Dropout
    '''
    model = keras.Sequential()
    model.add(layers.Dense(200, input_dim=9, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(layers.Dense(6, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    y = pd.get_dummies(train_Y.playlist_genre)
    tst_Y = pd.get_dummies(test_Y.playlist_genre)
    val_Y = pd.get_dummies(valid_Y.playlist_genre)

    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)

    training = model.fit(train_X, y, epochs=250, batch_size=200, validation_data=(valid_X, val_Y), callbacks=[stop])
    pred_Y = model.predict(test_X)

    plot_result(training)
    print(metrics.classification_report(not_normalized_test_Y, np.argmax(pred_Y, axis=1)))
    '''

    # Uloha 3 - Batch normalization
    '''
    model = keras.Sequential()
    model.add(layers.Dense(200, input_dim=9, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(layers.Dense(100, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(layers.Dense(6, activation='sigmoid'))

    optimizer = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    y = pd.get_dummies(train_Y.playlist_genre)
    tst_Y = pd.get_dummies(test_Y.playlist_genre)
    val_Y = pd.get_dummies(valid_Y.playlist_genre)

    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000)

    training = model.fit(train_X, y, epochs=250, batch_size=200, validation_data=(valid_X, val_Y), callbacks=[stop])
    pred_Y = model.predict(test_X)

    plot_result(training)
    print(metrics.classification_report(not_normalized_test_Y, np.argmax(pred_Y, axis=1)))
    '''

    # Uloha 4 - SVM klasifikator
    '''
    c_range = [1, 10, 100, 1000]
    gama_range = [1, 0.1, 0.001, 0.0001]
    param_grid = dict(gamma=gama_range, C=c_range)
    grid = GridSearchCV(SVC(verbose=1), param_grid=param_grid, verbose=1)

    grid.fit(train_X, not_normalized_train_Y.values.ravel())
    scores = grid.cv_results_['mean_test_score'].reshape(len(c_range), len(gama_range))
    pred_Y = grid.predict(test_X)

    print("Params %s Score %0.2f" % (grid.best_params_, grid.best_score_))
    print(metrics.classification_report(not_normalized_test_Y, pred_Y))
    '''