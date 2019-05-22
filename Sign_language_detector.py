from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

np.set_printoptions(threshold=np.inf)
optimizer = "Adam"
loss_func = "categorical_crossentropy"
#loss_func = "kullback_leibler_divergence"

class SignLangAgent:
    NUMBER_OF_CHARS = 29

    def init_model(self, input_shape, output_shape):
        print("----------------INITIALIZING MODEL----------------")
        start_time_for_init = datetime.datetime.now()
        model = Sequential()
        model.add(Dense(100, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation="softmax"))
        model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])
        self.model = model
        end_time_for_init = datetime.datetime.now()
        time_elapsed =  end_time_for_init - start_time_for_init
        print("------------FINISH INITIALIZING MODEL-------------")
        print("--------TIME TAKEN FOR INIT: {}-------".format(time_elapsed))

    def init_cnn_model(self, input_shape, output_shape):
        print("--------------INITIALIZING CNN MODEL--------------")
        start_time_for_init = datetime.datetime.now()
        
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="linear", input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="linear"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="linear"))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation="softmax"))
        model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])
        self.model = model
        """
        model = Sequential()
        model.add(Conv2D(16, (2,2), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(32, (3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
        model.add(Conv2D(64, (5,5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_shape, activation='softmax'))
        """
        self.model = model
        model.compile(optimizer=optimizer, loss=loss_func, metrics=["accuracy"])
        end_time_for_init = datetime.datetime.now()
        time_elapsed =  end_time_for_init - start_time_for_init
        print("----------FINISH INITIALIZING CNN MODEL-----------")
        print("--------TIME TAKEN FOR INIT: {}-------".format(time_elapsed))
        
    def train_model(self, x_train, y_train, validation_split=0.3, epochs=10, patience=3):
        print("---------------TRAINING IN PROGRESS---------------")
        early_stopping = EarlyStopping(patience=patience)
        return self.model.fit(x_train, y_train, validation_split=validation_split, epochs=epochs, callbacks=[early_stopping])

    def predict(self, x):
        print("--------------------PREDICTING--------------------")
        return self.model.predict(x)

    def convert_prediction_array_to_output(self, prediction):
        index = np.argmax(prediction)
        ascii_of_letter = index + 65
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

    def convert_img_to_numpy_arr(self, image_dir):
        gray_img = cv2.imread(image_dir, 0)
        np_img = np.array(gray_img)
        return np_img

    def obtain_testdata(self, dir_path):
        dict_of_dir = {'del' : [], 'nothing' : [], 'space' : []}
        for ascii_of_letter in range(65, 91):
            dict_of_dir[chr(ascii_of_letter)] = []

        for img_dir in os.listdir(dir_path):
            key = ""
            for letter in img_dir:
                if not letter.isdigit():
                    key += letter
                else:
                    break
            full_img_dir = os.path.join(dir_path, img_dir)
            dict_of_dir[key].append(full_img_dir)
        return dict_of_dir

    def save_model(self):
        print("-------------------SAVING MODEL-------------------")
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")
        print("-------------MODEL SUCCESSFULLY SAVED-------------")

    def load_model(self):
        print("------------------LOADING MODEL-------------------")
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("------------MODEL SUCCESSFULLY LOADED-------------")

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

        
if __name__ == "__main__":
    #Draws the images in x_train
    def check_images(x_train):
        temp = x_train.reshape((-1, 64, 64))
        for x in temp:
            plt.imshow(x, interpolation="nearest")
            plt.show()

    #Asks input from user for the type of neural network to use
    user_input = input("Use normal neural network(N) or \
convolutional neural network(C)? ")
    while user_input != 'N' and user_input != 'C':
        user_input = input("Use normal neural network(N) or \
convolutional neural network(C)? ")

    Agent = SignLangAgent()
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),\
                            "data_set_formatted")
    dict_of_dir = Agent.obtain_testdata(dir_path)

    LIST_OF_ASCII = []
    for ascii_of_letter in range(65, 91):
        LIST_OF_ASCII.append(chr(ascii_of_letter))
    LIST_OF_ASCII = LIST_OF_ASCII + ['del', 'nothing', 'space']
    img_dim = len(Agent.convert_img_to_numpy_arr(\
        dict_of_dir[LIST_OF_ASCII[0]][0]))

    start_time_for_load = datetime.datetime.now()
    print("-----------------LOADING IMAGES-------------------")
    
    len_of_values = 0
    for value in dict_of_dir.values():
        len_of_values += len(value)
    x_train = np.zeros((len_of_values, img_dim ** 2))
    y_train = np.zeros((len_of_values, len(dict_of_dir)))
    value_count = 0
    for key, values in dict_of_dir.items():
        for value in values:
            np_img = Agent.convert_img_to_numpy_arr(value).flatten()
            x_train[value_count] = np_img
            alphabet_index = LIST_OF_ASCII.index(key)
            temp_y_train = np.zeros((Agent.NUMBER_OF_CHARS,), dtype=int)
            temp_y_train[alphabet_index] = 1
            y_train[value_count] = temp_y_train
            value_count += 1
            if value_count % 500 == 0:
                print("----------------LOADED {} IMAGES".format(value_count)\
                      + "-" * (20 - len(str(value_count))))

    '''
    x_train = []
    y_train = []
    value_count = 0
    for key, values in dict_of_dir.items():
        for value in values:
            np_img = Agent.convert_img_to_numpy_arr(value).flatten()
            x_train = x_train + np_img.tolist()
            alphabet_index = LIST_OF_ASCII.index(key)
            temp_y_train = [0] * len(LIST_OF_ASCII)
            temp_y_train[alphabet_index] = 1
            y_train = y_train + temp_y_train
            value_count += 1
            if value_count % 50 == 0:
                print("----------------LOADED {} IMAGES----------------".format(value_count))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    '''
    
    end_time_for_load = datetime.datetime.now()
    time_elapsed =  end_time_for_load - start_time_for_load
    print("--------------FINISH LOADING IMAGES---------------")
    print("--------TIME TAKEN FOR LOAD: {}-------".format(time_elapsed))

    y_train = y_train.reshape(-1, len(LIST_OF_ASCII))
    if user_input == 'N':
        x_train = x_train.reshape(-1, img_dim ** 2)
        Agent.init_model((img_dim ** 2,), len(dict_of_dir))
    else:
        x_train = x_train.reshape(-1, img_dim, img_dim, 1)
        Agent.init_cnn_model((img_dim, img_dim, 1), len(dict_of_dir))
    normalise = 255
    x_train = x_train.astype('float32') / normalise
    y_train = y_train.astype('float32') / normalise
    history = Agent.train_model(x_train, y_train)
    Agent.plot_history(history)

    Agent.save_model()
