from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

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

    def init_model(self, input_shape, output_shape):
        print("----------------INITIALIZING MODEL----------------")
        start_time_for_init = datetime.datetime.now()
        model = Sequential()
        model.add(Dense(100, activation="relu", input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation="softmax"))
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
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
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=(3, 3), activation="linear", input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="linear"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation="linear"))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(output_shape, activation="softmax"))
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        end_time_for_init = datetime.datetime.now()
        time_elapsed =  end_time_for_init - start_time_for_init
        print("----------FINISH INITIALIZING CNN MODEL-----------")
        print("--------TIME TAKEN FOR INIT: {}-------".format(time_elapsed))
        
    def train_model(self, x_train, y_train, validation_data=None, validation_split=0.3, epochs=1000, patience=50):
        print("---------------TRAINING IN PROGRESS---------------")
        early_stopping = EarlyStopping(patience=patience)
        folderpath = "mnist_checkpoint"
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
        filepath = os.path.join(folderpath, "mnist_model_weights.h5")
        checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
        if validation_data == None:
            return self.model.fit(x_train, y_train,\
                                  validation_split=validation_split,\
                                  epochs=epochs, callbacks=[early_stopping, checkpoint])
        else:
            return self.model.fit(x_train, y_train,\
                                  validation_data=validation_data,\
                                  epochs=epochs, callbacks=[early_stopping, checkpoint])


    def predict(self, x):
        print("--------------------PREDICTING--------------------")
        return self.model.predict(x)

    def convert_prediction_array_to_output(self, prediction):
        index = np.argmax(prediction)
        output = self.LIST_OF_ASCII[index]
        return output

    def obtain_img_data(self, dir_path):
        #dict_of_img stores the correct prediction as the key
        #and the np array of all the images with the same
        #prediction as the value to the corresponding key
        dict_of_img = {}
        for key in self.LIST_OF_ASCII:
            dict_of_img[key] = []

        for key in os.listdir(dir_path):
            for img_dir in os.listdir(os.path.join(dir_path, key)):
                full_img_dir = os.path.join(dir_path, key, img_dir)
                dict_of_img[key].append(full_img_dir)
        for key, values in dict_of_img.items():
            temp = []
            for value in values:
                gray_img = cv2.imread(value, 0)
                resized_img = cv2.resize(gray_img, (64, 64), \
                                         interpolation = cv2.INTER_AREA)
                np_img = np.array(resized_img)
                temp.append(np_img)
            dict_of_img[key] = temp

        
    def obtain_csv_data(self, dir_path, data_type):
        #dict_of_img stores the correct prediction as the key
        #and the np array of all the images with the same
        #prediction as the value to the corresponding key
        dict_of_img = {}
        for key in self.LIST_OF_ASCII:
            dict_of_img[key] = []
            
        for excel in os.listdir(dir_path):
            if data_type in excel:
                train_data = pd.read_csv(os.path.join(dir_path, excel)).values
            '''
            else:
                test_data = pd.read_csv(os.path.join(dir_path, excel),\
                                         skiprows=1).values)
            '''
        train_x, train_y = train_data[:, 1:], train_data[:, 0]
        for index in range(len(train_y)):
            key = self.LIST_OF_ASCII[train_y[index]]
            dict_of_img[key].append(train_x[index].reshape((28, 28)))
            
        return dict_of_img

    def save_model(self):
        print("-------------------SAVING MODEL-------------------")
        model_json = self.model.to_json()
        #save model as model.json
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        #save weights as model.h5
        self.model.save_weights("model.h5")
        print("-------------MODEL SUCCESSFULLY SAVED-------------")

    def load_model(self):
        print("------------------LOADING MODEL-------------------")
        #load model from model.json
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights from model.h5
        loaded_model.load_weights("model.h5")
        self.model = loaded_model
        print("------------MODEL SUCCESSFULLY LOADED-------------")

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            self.model.load_weights(checkpoint_path)
        else:
            print("Checkpoint not found")

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
    have_validation_data = True
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
                            "data_set_mnist")
    dict_of_img = Agent.obtain_csv_data(dir_path, 'train')
    
    img_dim = len(dict_of_img[Agent.LIST_OF_ASCII[0]][0])

    start_time_for_load = datetime.datetime.now()
    print("-----------------LOADING IMAGES-------------------")
    
    len_of_values = 0
    for value in dict_of_img.values():
        len_of_values += len(value)
    x_train = np.zeros((len_of_values, img_dim ** 2))
    y_train = np.zeros((len_of_values, len(dict_of_img)))
    value_count = 0
    for key, values in dict_of_img.items():
        for value in values:
            np_img = value.flatten()
            x_train[value_count] = np_img
            alphabet_index = Agent.LIST_OF_ASCII.index(key)
            temp_y_train = np.zeros((len(Agent.LIST_OF_ASCII),), dtype=int)
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
    for key, values in dict_of_img.items():
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

    y_train = y_train.reshape(-1, len(Agent.LIST_OF_ASCII))

    if have_validation_data:
        dict_of_img = Agent.obtain_csv_data(dir_path, 'test')
        
        img_dim = len(dict_of_img[Agent.LIST_OF_ASCII[0]][0])
        
        len_of_test_values = 0
        for value in dict_of_img.values():
            len_of_test_values += len(value)
        x_test = np.zeros((len_of_test_values, img_dim ** 2))
        y_test = np.zeros((len_of_test_values, len(dict_of_img)))
        value_count = 0
        for key, values in dict_of_img.items():
            for value in values:
                np_img = value.flatten()
                x_test[value_count] = np_img
                alphabet_index = Agent.LIST_OF_ASCII.index(key)
                temp_y_test = np.zeros((len(Agent.LIST_OF_ASCII),), dtype=int)
                temp_y_test[alphabet_index] = 1
                y_test[value_count] = temp_y_test
                value_count += 1
    
    if user_input == 'N':
        if have_validation_data:
            x_test = x_test.reshape(-1, img_dim ** 2)
        x_train = x_train.reshape(-1, img_dim ** 2)
        Agent.init_model((img_dim ** 2,), len(dict_of_img))
    else:
        if have_validation_data:
            x_test = x_test.reshape(-1, img_dim, img_dim, 1)
        x_train = x_train.reshape(-1, img_dim, img_dim, 1)
        Agent.init_cnn_model((img_dim, img_dim, 1), len(dict_of_img))
    if have_validation_data:
        x_test = x_test.astype('float32') / 255
        y_test = y_test.astype('float32') / 255
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('float32') / 255
    Agent.load_checkpoint(os.path.join(os.path.dirname(os.path.abspath(__file__)),\
                                       "mnist_checkpoint", "mnist_model_weights.h5"))
    if have_validation_data:
        history = Agent.train_model(x_train, y_train, \
                                    validation_data=(x_test, y_test),\
                                    epochs=100, patience=50)
    else:
        history = Agent.train_model(x_train, y_train, epochs=1000, patience=50)
    Agent.plot_history(history)

    Agent.save_model()
