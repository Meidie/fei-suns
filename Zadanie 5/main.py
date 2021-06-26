import numpy as np
import pandas as pd
import plotly as plotly
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from os import path
import datetime
# Neslo mi to takto priamo importovat, musel som to urobit cez terminal -> pip install pillow
#from PIL import Image


def create_test_data(filenames, df):
    for i, s in enumerate(filenames):
        filenames[i] = s.replace('.jpg', '')

    df = df[(df.id.isin(filenames))]
    df = df.sort_values(by=['id'])
    return df


def delete_unmatched_images(df):
    for filePath in df.path:
        if not path.exists('images/' + filePath):
            df = df[df.path != filePath]

    return df


def delete_invalid_records(labels):
    number_of_unique_values = labels.masterCategory.value_counts()
    return labels.drop(labels[(labels.masterCategory == 'Home') | (labels.masterCategory == 'Sporting Goods')].index)


def load_images():
    labels = pd.read_csv('styles.csv', usecols=['id', 'masterCategory'], encoding="utf-8")
    labels = delete_invalid_records(labels)
    labels['path'] = labels.apply(lambda row: str(row['id']) + '.jpg', axis=1)
    labels = delete_unmatched_images(labels)

    image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=0.2)

    training_generator = image_generator.flow_from_dataframe(
        dataframe=labels,
        directory='images',
        x_col='path',
        y_col='masterCategory',
        shuffle=True,
        target_size=(40, 40),
        batch_size=100,
        subset='training'
    )

    validation_generator = image_generator.flow_from_dataframe(
        dataframe=labels,
        directory='images',
        x_col='path',
        y_col='masterCategory',
        shuffle=True,
        target_size=(40, 40),
        batch_size=100,
        subset='validation'
    )

    # Batch size - cislo delitelne poctom zaznamov
    test_data = create_test_data(validation_generator.filenames, labels)
    test_data = test_data.head(8000)
    test_image_generator= ImageDataGenerator(rescale=1. / 255.)
    test_generator = test_image_generator.flow_from_dataframe(
        dataframe=test_data,
        directory='images',
        x_col='path',
        y_col='masterCategory',
        shuffle=False,
        target_size=(40, 40),
        batch_size=80,
    )

    return training_generator, validation_generator, test_generator


def graph_evaluation(training_data):
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


def graph_evaluation_from_csv(csv_data):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
    fig.add_trace(go.Scatter(x=csv_data.epoch, y=csv_data.accuracy,
                             name="Training accuracy", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=csv_data.epoch, y=csv_data.val_accuracy,
                             name="Validation accuracy", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=csv_data.epoch, y=csv_data.loss,
                             name="Training loss", mode="lines"), row=1, col=2)
    fig.add_trace(go.Scatter(x=csv_data.epoch, y=csv_data.val_loss,
                             name="Validation loss", mode="lines"), row=1, col=2)

    plotly.offline.plot(fig, filename='result_plot.html')


def plot_confusion_matrix(cm):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.show()


def train(training_generator, validation_generator, test_generator):

    # CVSLogger callback
    csv_log = CSVLogger('results.csv')

    # Tensorboard callback
    log_dir = 'logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Checkpoint callback
    checkpoint_filepath = 'checkpoints/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    stopping = EarlyStopping(monitor='val_loss', patience=5)
    optimizer = optimizers.Adam(lr=0.0001)

    # vstup -> vystup
    model = Sequential()
    model.add(layers.Conv2D(40, (3, 3), padding='same', input_shape=(40, 40, 3)))

    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(40, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(60, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    # vystupna vrstva
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    STEP_SIZE_TRAIN = training_generator.n // training_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    # Nacitanie checkpoint modelu
    # model.load_weights('checkpoints/20201208-183610')

    training = model.fit(
        training_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=25,
        callbacks=[stopping, tensorboard, checkpoint, csv_log],
        verbose=True
    )

    eval = model.evaluate(
        validation_generator,
        steps=STEP_SIZE_VALID)

    print(eval)

    test_generator.reset()
    pred = model.predict(
        test_generator,
        steps=STEP_SIZE_TEST)

    predicted_classes = np.argmax(pred, axis=1)
    classes = test_generator.classes

    cm = confusion_matrix(classes, predicted_classes)
    print(cm)
    plot_confusion_matrix(cm)
    print(classification_report(classes, predicted_classes))

    # V pripade pouzitia checkpoint modelu -> na nacitanie dat trenovania + vykresleenie priebehu na graf
    # results = pd.read_csv('results.csv')
    # graph_evaluation_from_csv(results)

    graph_evaluation(training)


if __name__ == '__main__':

    train_gen, val_gen, test_gen = load_images()
    train(train_gen, val_gen, test_gen)
    print('DONE')
