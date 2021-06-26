import numpy as np
import pandas as pd
import seaborn as sns
import plotly as plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor


def read_csv(name):
    return pd.read_csv(name, index_col=0, encoding="utf-8")


def empty_val_check(df):
    print(df.isna().sum())


def clear_unused_columns(df):
    cols = ['u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field', 'plate', 'mjd', 'fiberid', 'class', 'x_coord', 'y_coord',
            'z_coord']
    return df[cols]


def replace_string_values(df):
    df = df.replace("STAR", 0)
    df = df.replace("QSO", 1)
    df = df.replace("GALAXY", 2)

    return df


def clear_wrong_data(df, columns):

    fig = make_subplots(rows=4, cols=3, subplot_titles=list(map(lambda x: x + ':class', columns)))
    fig.add_trace(go.Scatter(x=df[columns[0]], y=df['class'], mode="markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df[columns[1]], y=df['class'], mode="markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=df[columns[2]], y=df['class'], mode="markers"), row=1, col=3)
    fig.add_trace(go.Scatter(x=df[columns[3]], y=df['class'], mode="markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df[columns[4]], y=df['class'], mode="markers"), row=2, col=2)
    fig.add_trace(go.Scatter(x=df[columns[5]], y=df['class'], mode="markers"), row=2, col=3)
    fig.add_trace(go.Scatter(x=df[columns[6]], y=df['class'], mode="markers"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df[columns[7]], y=df['class'], mode="markers"), row=3, col=2)
    fig.add_trace(go.Scatter(x=df[columns[8]], y=df['class'], mode="markers"), row=3, col=3)
    fig.add_trace(go.Scatter(x=df[columns[9]], y=df['class'], mode="markers"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df[columns[10]], y=df['class'], mode="markers"), row=4, col=2)

    plotly.offline.plot(fig, filename='plot.html')

    return df


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


def split(df_X, df_Y):
    X_train, X_valid, y_train, y_valid = train_test_split(df_X, df_Y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid


def tree_to_png(estimator, fn, cn):
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(40, 2), dpi=900)

    for index in range(0, 5):
        tree.plot_tree(estimator.estimators_[index], feature_names=fn, class_names=cn, filled=True, ax=axes[index])
        axes[index].set_title('Estimator: ' + str(index), fontsize=5)

    fig.savefig('rf_5trees.png')


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


def evaluate_regression_results(test_Y, pred_Y):
    print("Mean squared error - X: %.5f" % mean_squared_error(test_Y['x_coord'], pred_Y[:, 0]))
    print("Mean squared error - Y: %.5f" % mean_squared_error(test_Y['y_coord'], pred_Y[:, 1]))
    print("Mean squared error - Z: %.5f" % mean_squared_error(test_Y['z_coord'], pred_Y[:, 2]))
    print("Mean squared error - TOTAL: %.5f" % mean_squared_error(test_Y, pred_Y))

    print("Coefficient of determination - X: %.5f" % r2_score(test_Y['x_coord'], pred_Y[:, 0]))
    print("Coefficient of determination - Y: %.5f" % r2_score(test_Y['y_coord'], pred_Y[:, 1]))
    print("Coefficient of determination - Z: %.5f" % r2_score(test_Y['z_coord'], pred_Y[:, 2]))
    print("Coefficient of determination - TOTAL: %.5f" % r2_score(test_Y, pred_Y))


def evaluate_crossval(model, valid_X, valid_Y):
    result = cross_val_score(model, valid_X, valid_Y, cv=10, n_jobs=-1)
    print("Max acc : " + str(result.max()))
    print("Min acc : " + str(result.min()))
    print("avg acc : " + str(result.mean()))


def init_classification_data():
    test_data = read_csv("test.csv")
    train_data = read_csv("train.csv")

    test_data = clear_unused_columns(replace_string_values(test_data))
    train_data = clear_unused_columns(replace_string_values(train_data))

    columns = ['u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field', 'plate', 'mjd', 'fiberid', 'x_coord', 'y_coord',
               'z_coord', 'class']
    X_columns = ['u', 'g', 'i', 'z', 'run', 'field', 'plate', 'mjd', 'x_coord', 'z_coord']

    # clear_wrong_data(train_data, X_columns)
    heat_map(train_data[columns])

    train_X, valid_X, train_Y, valid_Y = split(train_data[X_columns], train_data[['class']])
    test_X = test_data[X_columns]
    test_Y = test_data[['class']]

    return train_X, valid_X, test_X, train_Y, valid_Y, test_Y;


def init_regression_data():
    test_data = read_csv("test.csv")
    train_data = read_csv("train.csv")

    test_data = clear_unused_columns(test_data)
    train_data = clear_unused_columns(train_data)

    test_data = pd.concat([test_data.drop('class', axis=1), pd.get_dummies(test_data['class'])], axis=1)
    train_data = pd.concat([train_data.drop('class', axis=1), pd.get_dummies(train_data['class'])], axis=1)

    columns = ['u', 'g', 'r', 'i', 'z', 'run', 'camcol', 'field', 'plate', 'mjd', 'fiberid', 'x_coord', 'y_coord',
               'z_coord', 'STAR', 'QSO', 'GALAXY']
    X_columns = ['u', 'g', 'r', 'i', 'z', 'plate', 'mjd', 'fiberid', 'STAR', 'QSO', 'GALAXY']
    heat_map(train_data[columns])

    train_X, valid_X, train_Y, valid_Y = split(train_data[X_columns], train_data[['x_coord', 'y_coord', 'z_coord']])

    test_X = test_data[X_columns]
    test_Y = test_data[['x_coord', 'y_coord', 'z_coord']]

    return train_X, valid_X, test_X, train_Y, valid_Y, test_Y


def random_forest_classifier(train_X, valid_X, test_X, train_Y, valid_Y, test_Y):
    print("\n##### Random Forest Classifier #####")
    classifier = RandomForestClassifier(max_depth=3, n_estimators=15, random_state=38)
    classifier.fit(train_X, train_Y.values.ravel())
    pred_Y = classifier.predict(test_X)

    print(metrics.classification_report(test_Y, pred_Y))

    evaluate_crossval(classifier, valid_X, valid_Y.values.ravel())

    class_labels = ["STAR", "QSO", "GALAXY"]
    tree_to_png(classifier, test_X.columns.values, class_labels)


def mlp_classifier(train_X, valid_X, test_X, train_Y, valid_Y, test_Y):
    print("\n##### MLPClassifier #####")

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    valid_X = scaler.transform(valid_X)

    classifier = MLPClassifier(hidden_layer_sizes=(20,), alpha=0.01, tol=0.001, solver="adam", random_state=10,
                               verbose=False, max_iter=1000)
    classifier.fit(train_X, train_Y.values.ravel())
    y_pred_nn = classifier.predict(test_X)
    final_score = classifier.score(test_X, test_Y)

    print("Final score: %.5f" % final_score)
    print(confusion_matrix(test_Y, y_pred_nn))

    plot_confusion_matrix(classifier, test_X, test_Y)
    plt.show()

    print(metrics.classification_report(test_Y, y_pred_nn))
    evaluate_crossval(classifier, valid_X, valid_Y.values.ravel())


def random_forest_regression(train_X, valid_X, test_X, train_Y, valid_Y, test_Y):
    print("\n##### Random Forest Regressor #####")
    regressor = RandomForestRegressor(n_estimators=15, random_state=38, verbose=False)
    regressor.fit(train_X, train_Y)
    pred_Y = regressor.predict(test_X)

    final_score = regressor.score(test_X, test_Y)
    print("Final score: %.5f" % final_score)

    class_labels = []
    tree_to_png(regressor, test_X.columns.values, class_labels)
    evaluate_regression_results(test_Y, pred_Y)
    evaluate_crossval(regressor, valid_X, valid_Y)


def mlp_regressor(train_X, valid_X, test_X, train_Y, valid_Y, test_Y):
    print("\n##### MLPRegressor #####")

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)
    valid_X = scaler.transform(valid_X)

    regressor = MLPRegressor(hidden_layer_sizes=(50,), learning_rate="constant", alpha=0.001, tol=0.001,
                             random_state=1, verbose=False, max_iter=10000)
    regressor.fit(train_X, train_Y)
    pred_Y = regressor.predict(test_X)
    final_score = regressor.score(test_X, test_Y)

    print("Final score: %.5f" % final_score)
    evaluate_regression_results(test_Y, pred_Y)


def decision_tree_regressor(train_X, valid_X, test_X, train_Y, valid_Y, test_Y):
    print("\n##### DecisionTreeRegressor #####")

    model = DecisionTreeRegressor()
    model.fit(train_X, train_Y)
    pred_Y = model.predict(test_X)
    final_score = model.score(test_X, test_Y)

    print("Final score: %.5f" % final_score)
    evaluate_regression_results(test_Y, pred_Y)


if __name__ == '__main__':

    # Klasifikacia
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = init_classification_data()
    random_forest_classifier(train_X, valid_X, test_X, train_Y, valid_Y, test_Y)
    # mlp_classifier(train_X, valid_X, test_X, train_Y, valid_Y, test_Y)

    # Regresia
    # train_X, valid_X, test_X, train_Y, valid_Y, test_Y = init_regression_data()
    # random_forest_regression(train_X, valid_X, test_X, train_Y, valid_Y, test_Y)
    # mlp_regressor(train_X, valid_X, test_X, train_Y, valid_Y, test_Y)
    # decision_tree_regressor(train_X, valid_X, test_X, train_Y, valid_Y, test_Y)

    print("\nDONE")
