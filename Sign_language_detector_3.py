from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

np.set_printoptions(threshold=np.inf)


class SignLangAgent:
    LIST_OF_ASCII = []
    for ascii_of_letter in range(65, 91):
        LIST_OF_ASCII.append(chr(ascii_of_letter))
    LIST_OF_ASCII = LIST_OF_ASCII + ['del', 'nothing', 'space']

    def __init__(self, WIDTH, HEIGHT, BATCH_SIZE=8):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.BATCH_SIZE = BATCH_SIZE

    def init_model(self, input_shape, output_shape):
        print("----------------INITIALIZING MODEL----------------")
        start_time_for_init = datetime.datetime.now()
        model = Sequential()
        model.add(Dense(100, activation="relu", input_shape=input_shape))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(output_shape, activation="softmax"))
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        end_time_for_init = datetime.datetime.now()
        time_elapsed = end_time_for_init - start_time_for_init
        print("------------FINISH INITIALIZING MODEL-------------")
        print("--------TIME TAKEN FOR INIT: {}-------".format(time_elapsed))

    def init_cnn_model(self, input_shape, num_classes):
        print("--------------INITIALIZING CNN MODEL--------------")
        start_time_for_init = datetime.datetime.now()
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)

        predictions = Dense(num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        adam = Adam(lr=0.00001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

        end_time_for_init = datetime.datetime.now()
        time_elapsed = end_time_for_init - start_time_for_init
        print("----------FINISH INITIALIZING CNN MODEL-----------")
        print("--------TIME TAKEN FOR INIT: {}-------".format(time_elapsed))

    def train_model(self, train_generator, num_train_images, epochs=10, patience=3):
        print("---------------TRAINING IN PROGRESS---------------")
        folderpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),\
                                  "ResNet50_checkpoint")
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        filepath = os.path.join(folderpath, "ResNet50_model_weights.h5")
        earlyStopping = EarlyStopping(patience=patience)
        checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
        callbacks_list = [checkpoint, earlyStopping]
        return self.model.fit_generator(train_generator, epochs=epochs, workers=8,
                                       steps_per_epoch=num_train_images // self.BATCH_SIZE,
                                       shuffle=True, callbacks=callbacks_list)

    def predict(self, x):
        print("--------------------PREDICTING--------------------")
        return self.model.predict(x)

    def convert_prediction_array_to_output(self, prediction):
        index = np.argmax(prediction)
        ascii_of_letter = index + 65
        # ASCII 65 = 'A' and ASCII 90 = 'Z'
        if 65 <= ascii_of_letter <= 90:
            output = chr(ascii_of_letter)
        else:
            if ascii_of_letter == 91:
                output = 'del'
            elif ascii_of_letter == 92:
                output = 'nothing'
            elif ascii_of_letter == 93:
                output = 'space'
        return output

    def obtain_testdata(self, train_dir):
        print("------------------OBTAINING DATA------------------")
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            horizontal_flip=True,
            vertical_flip=True
        )

        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(self.HEIGHT, self.WIDTH),
                                                            batch_size=self.BATCH_SIZE)
        print("--------------DATA SUCCESSFULLY SAVED-------------")
        return train_generator

    def save_model(self):
        print("-------------------SAVING MODEL-------------------")
        model_json = self.model.to_json()
        # save model as model.json
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # save weights as model.h5
        self.model.save_weights("model.h5")
        print("-------------MODEL SUCCESSFULLY SAVED-------------")

    def load_model(self):
        print("------------------LOADING MODEL-------------------")
        # load model from model.json
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights from model.h5
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("------------MODEL SUCCESSFULLY LOADED-------------")

    def load_checkpoint(self, checkpoint_path):
        print("----------------LOADING CHECKPOINT----------------")
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
            print("----------CHECKPOINT SUCCESSFULLY LOADED----------")
        else:
            print("---------------CHECKPOINT NOT FOUND!--------------")

    def plot_history(self, history):
        print("-----------------PLOTTING RESULTS-----------------")
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.savefig('acc_vs_epochs.png')


if __name__ == "__main__":
    IMG_DIM = 200
    NUM_OF_SAMPLES = 8724
    
    # Draws the images in x_train
    def check_images(x_train):
        temp = x_train.reshape((-1, 200, 200))
        for x in temp:
            plt.imshow(x, interpolation="nearest")
            plt.show()

    Agent = SignLangAgent(IMG_DIM, IMG_DIM)
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),\
                            "data_set")
    train_generator = Agent.obtain_testdata(dir_path)

    Agent.init_cnn_model((IMG_DIM, IMG_DIM, 3), len(Agent.LIST_OF_ASCII))
    Agent.load_checkpoint(os.path.join(os.path.dirname(os.path.abspath(__file__)),\
                                       "ResNet50_checkpoint", "ResNet50_model_weights.h5"))
    history = Agent.train_model(train_generator, NUM_OF_SAMPLES)
    Agent.plot_history(history)

    Agent.save_model()
