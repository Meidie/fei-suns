import pandas as pd
from collections import OrderedDict
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly as plotly
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot
from sklearn.linear_model import Ridge


def read_csv():
    return pd.read_csv("srdcove_choroby.csv", index_col=0, encoding="utf-8")


def remove_empty(df):
    return df.dropna()


def replace_string_values(df):
    df = df.replace("normal", 0)
    df = df.replace("above normal", 1)
    df = df.replace("well above normal", 2)
    df = df.replace("man", 0)
    df = df.replace("woman", 1)

    return df


def clear_wrong_data(df):
    # odstranenie zaznamov s nezmyselnym vekom
    df = df.drop(df[df['age'] > 36500].index)
    # odstranenie zaznamov s nezmyselnym systolickym tlakom
    df = df.drop(df[(df['ap_hi'] >= 210) | (df['ap_hi'] < 50)].index)
    # odstranenie zaznamov s nezmyselnym diastolickym tlakom
    df = df.drop(df[(df['ap_lo'] >= 120) | (df['ap_lo'] < 40)].index)

    # fig = make_subplots(rows=2, cols=3, subplot_titles=("Age:Target", "Height:Target", "Weight:Target", "ap_hi:Target", "ap_lo:Target"))
    # fig.add_trace(go.Scatter(x=df['age'], y=df['cardio'], mode="markers"), row=1, col=1)
    # fig.add_trace(go.Scatter(x=df['height'], y=df['cardio'], mode="markers"), row=1, col=2)
    # fig.add_trace(go.Scatter(x=df['weight'], y=df['cardio'], mode="markers"), row=1, col=3)
    # fig.add_trace(go.Scatter(x=df['ap_hi'], y=df['cardio'], mode="markers"), row=2, col=1)
    # fig.add_trace(go.Scatter(x=df['ap_lo'], y=df['cardio'], mode="markers"), row=2, col=2)

    # dff = df.apply(lambda x: True
    # if (x['smoke'] == 1 and x['age'] < 5500) else False, axis=1)
    # num_rows = len(dff[dff == True].index)
    # print('Number of Rows: ', num_rows)

    #plotly.offline.plot(fig, filename='plot.html')

    return df


def correlation(df, col):
    cc_dictionary = {}

    for column in df:
        correlation_matrix = r = np.corrcoef(df[col], df[column])
        correlation_coefficient = correlation_matrix[0][1]
        cc_dictionary[column + '_cc'] = correlation_coefficient

    ordered_cc_dictionary = OrderedDict(sorted(cc_dictionary.items(), key=itemgetter(1), reverse=True))

    for entry in ordered_cc_dictionary:
        print(entry + ": " + str(ordered_cc_dictionary[entry]))


def split_data(df):
    x = df.drop(['cardio', 'ap_hi'], axis=1)
    y_cardio = df['cardio']
    y_ap_hi = df['ap_hi']
    return x, y_cardio, y_ap_hi


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


def convertDataTypes(df):
    return df.astype(float)


def split_sets_cardio(df):
    X = df.loc[:, df.columns != 'cardio']
    y = df.loc[:, df.columns == 'cardio']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def split_sets_bmi(df):
    X = df.loc[:, df.columns != 'bmi']
    y = df.loc[:, df.columns == 'bmi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def train_mpl_classifier(train_x, test_x, train_y, test_y):
    classifier = MLPClassifier(hidden_layer_sizes=(20,), alpha=0.001, tol=0.0001, random_state=1, verbose=False,
                             max_iter=1000)
    classifier.fit(train_x, train_y.values.ravel())
    y_pred_nn = classifier.predict(test_x)
    final_score = classifier.score(test_x, test_y)
    print(final_score)
    print(confusion_matrix(test_y, y_pred_nn))
    # plot_confusion_matrix(classifier, test_x, test_y)
    # plt.show()
    print(metrics.classification_report(test_y, y_pred_nn))


def train_mlp_regressor(train_x, test_x, train_y, test_y):
    classifier = MLPRegressor(hidden_layer_sizes=(20,), alpha=0.001, tol=0.0001, random_state=1, verbose=False,
                             max_iter=1000)
    classifier.fit(train_x, train_y.values.ravel())
    y_pred_nn = classifier.predict(test_x)
    # final_score = classifier.score(test_x, test_y)
    # print(final_score)
    print('\nMLPRegressor')
    print("Mean squared error: %.5f" % mean_squared_error(test_y, y_pred_nn))
    print("Coefficient of determination: %.5f" % r2_score(test_y, y_pred_nn))


def linear_regression(train_x, test_x, train_y, test_y):
    lin_reg = LinearRegression().fit(train_x, train_y)
    lin_reg.score(train_x, train_y)
    y_pred_nn = lin_reg.predict(test_x)
    # final_score = lin_reg.score(test_x, test_y)
    # print(final_score)
    print('\nLinear Regression')
    print('Mean squared error: %.5f' % mean_squared_error(test_y, y_pred_nn))
    print('Coefficient of determination: %.5f' % r2_score(test_y, y_pred_nn))


def calc_bmi(x):
    return x['weight'] / (x['height'] * x['height'] / 10000)


def add_bmi_column(df):
    df['bmi'] = df.apply(lambda x: calc_bmi(x), axis=1)


def create_bmi(df):
    bmi = df['bmi']
    df_without_height_weight = df.drop(columns=['height', 'weight', 'bmi'])
    return bmi, normalize(df_without_height_weight)


def split(df, Xcols, Y_col):
    X = df[Xcols]
    y = Y_col
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def heat_map(df):
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(
        df.corr(),
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(300, 400, n=200),
        square=True,
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        horizontalalignment='right'
    )
    plt.show()


def residual_plot(train_x, test_x, train_y, test_y):
    model = Ridge()
    visualizer = ResidualsPlot(model)

    visualizer.fit(train_x, train_y)
    visualizer.score(test_x, test_y)
    visualizer.show()


if __name__ == '__main__':
    '''1. Nacitajte data a pripravte ich na spracovanie '''
    # nacitanie dat
    data = read_csv()

    '''2. Data predspracujte'''
    # odstranenie prazdncyh hodnot
    data = remove_empty(data)
    # zakodovanie textovych hodnot
    data = replace_string_values(data)
    # odstranenie nezmyselnych hodnot
    data = clear_wrong_data(data)
    # data = convertDataTypes(data)
    # normalizacia
    n_data = normalize(data)
    # standardizacia
    s_data = scale(data)
    # rozdelenie na vstupne a vystupne parametre
    x, y_cardio, y_ap_hi = split_data(data)

    '''3. Analyzujte priznaky'''
    # korelacia
    correlation(data, 'cardio')
    # graf korelacie
    # heat_map(data)

    '''4. Natrenujte binarny klasifikator'''
    Xcols = ['age', 'weight', 'ap_hi', 'ap_lo', 'cholesterol']
    train_x, test_x, train_y, test_y = split(n_data, Xcols, y_cardio)
    train_mpl_classifier(train_x, test_x, train_y, test_y)

    '''5. Natrenujte regresor'''
    add_bmi_column(data)
    data = data.drop(data[(data['bmi'] >= 41) | (data['bmi'] <= 15)].index)

    Xcols = ['age', 'gender', 'ap_lo', 'ap_hi', 'cholesterol', 'cardio']
    y_bmi, df_without_height_weight = create_bmi(data)
    train_x, test_x, train_y, test_y = split(df_without_height_weight, Xcols, y_bmi)
    train_mlp_regressor(train_x, test_x, train_y, test_y)
    linear_regression(train_x, test_x, train_y, test_y)
    # residual_plot(train_x, test_x, train_y, test_y)
    # graf korelacie s bmi
    # heat_map(data)
