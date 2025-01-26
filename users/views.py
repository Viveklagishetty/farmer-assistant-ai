from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'crop_recommendation//Crop_recommendation.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def trainning(request):
    from django.conf import settings
    import numpy as np # linear algebra
    import pandas as pd  # data processing
    import os #  to interact with files using there paths
    from sklearn.datasets import load_files
    import tensorflow as tf
    import random
    import shutil

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import Dropout, MaxPooling2D, AveragePooling2D, Dense, Flatten, Input, Conv2D, add, Activation
    from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, Layer,
                          BatchNormalization, LocallyConnected2D,
                          ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose,
                          GaussianNoise, UpSampling2D, Input)

    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.models import Sequential , Model , load_model
    from tensorflow.keras.preprocessing.image import load_img , img_to_array , ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer

    from PIL import Image
    from tensorflow.keras.applications import MobileNetV2
    import matplotlib.pyplot as plt

    import cv2
    from imutils import paths
    import numpy as np
    import os
    import time
    from tensorflow.keras.utils import plot_model
    import warnings
    warnings.filterwarnings("ignore")

    print("Tensorflow version: ",tf.__version__)

    import requests
    import tensorflow as tf

    import tensorflow.keras.utils as np_utils

    access_token = '' #Access token here

    class botCallback(tf.keras.callbacks.Callback):
        def __init__(self,access_token):
            self.access_token = access_token
            self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/getUpdates'
            self.response = requests.get(self.ping_url).json()
            #print(self.response)
            self.chat_id = self.response['result'][0]['message']['chat']['id']
            #self.chat_id = self.response['result']

        def send_message(self,message):
            #print('sending message')
            self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendMessage?'+\
                        'chat_id='+str(self.chat_id)+\
                        '&parse_mode=Markdown'+\
                        '&text='+message
            self.response = requests.get(self.ping_url)
    
        def send_photo(self,filepath):
            imagefile= open(filepath,"rb")
            file_dict = {'photo':imagefile}
            self.ping_url = 'https://api.telegram.org/bot'+str(self.access_token)+'/sendPhoto?chat_id='+str(self.chat_id)
            self.response = requests.post(self.ping_url, files = file_dict)
            imagefile.close()

        def on_train_batch_begin(self, batch, logs=None):
            pass
    
        def on_train_batch_end(self, batch, logs=None):
            message = ' Iteration/Batch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(batch,logs['accuracy'],logs['loss'])
            #print(logs)
            try:
                message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
                self.send_message(message)
            except:
                pass

        def on_test_batch_begin(self, batch, logs=None):
            pass
    
        def on_test_batch_end(self, batch, logs=None):
            message = ' Iteration/Batch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(batch,logs['accuracy'],logs['loss'])
            try:
                message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
                self.send_message(message)
            except:
                pass

        def on_epoch_begin(self, epoch, logs=None):
            pass

        def on_epoch_end(self, epoch, logs=None):

            message = ' Epoch {}\n Training Accuracy : {:7.2f}\n Training Loss : {:7.2f}\n'.format(epoch,logs['accuracy'],logs['loss'])
            try:
                message += ' Validation Accuracy : {:7.2f}\n Validation Loss : {:7.2f}\n'.format(logs['val_accuracy'],logs['val_loss'])
                self.send_message(message)        
            except:
                pass

    class Plotter(botCallback):
        def __init__(self,access_token):
    
            super().__init__(access_token)
        def on_train_begin(self,logs=None):
            self.batch = 0
            self.epoch = []
            self.train_loss = []
            self.val_loss = []
            self.train_acc = []
            self.val_acc = []
            self.fig = plt.figure(figsize=(200,100))
            self.logs = []

        def on_epoch_end(self, epoch, logs=None):
            self.logs.append(logs)
            self.epoch.append(epoch)
            self.train_loss.append(logs['loss'])
            self.val_loss.append(logs['val_loss'])
            self.train_acc.append(logs['accuracy'])
            self.val_acc.append(logs['val_accuracy'])
            f,(ax1,ax2) = plt.subplots(1,2,sharex=True)
            #clear_output(wait=True)
            ax1.plot(self.epoch, self.train_loss, label='Training Loss')
            ax1.plot(self.epoch, self.val_loss, label='Validation Loss')
            ax1.legend()
            ax2.plot(self.epoch, self.train_acc, label='Training Accuracy')
            ax2.plot(self.epoch, self.val_acc, label='Validation Accuracy')
            ax2.legend()
            plt.savefig('Accuracy and Loss plot.jpg')
            self.send_photo('Accuracy and Loss plot.jpg')

            import os
            import random
            import shutil

    # Set paths to your soil image folders and the destination folders for train and test sets
    data_root = 'C:\\Users\\Admin\\Desktop\\New folder\\Soil types'
    train_root = 'C:\\Users\\Admin\\Desktop\\New folder\\train'
    test_root = 'C:\\Users\\Admin\\Desktop\\New folder\\test'

    # List of soil image folders
    soil_folders = ["Black Soil", "Cinder Soil", "Laterite Soil", "Peat Soil", "Yellow Soil"]

    # Split ratio (adjust as needed)
    split_ratio = 0.8  # 80% for training, 20% for testing

    # Iterate through each soil folder and split images into train and test sets
    for folder in soil_folders:
        folder_path = os.path.join(data_root, folder)
        train_folder = os.path.join(train_root, folder)
        test_folder = os.path.join(test_root, folder)

        # Create destination folders if they don't exist
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # List all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")]

        # Shuffle the image files
        random.shuffle(image_files)

        # Calculate the split index
        split_index = int(len(image_files) * split_ratio)

        # Split images into train and test sets
        train_images = image_files[:split_index]
        test_images = image_files[split_index:]

        # Move train images
        for img in train_images:
            src_path = os.path.join(folder_path, img)
            dest_path = os.path.join(train_folder, img)
            shutil.copy(src_path, dest_path)

        # Move test images
        for img in test_images:
            src_path = os.path.join(folder_path, img)
            dest_path = os.path.join(test_folder, img)
            shutil.copy(src_path, dest_path)

    print("Dataset splitting completed.")

    train_dir ='C:\\Users\\Admin\\Desktop\\New folder\\train'
    test_dir = 'C:\\Users\\Admin\\Desktop\\New folder\\test'

    image_size = 224

    batch_size = 32

    train_datagen = ImageDataGenerator(rescale = 1./255,
                            rotation_range=45,
                            zoom_range=0.40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.15,
                            horizontal_flip=True,
                            vertical_flip= True,
                            fill_mode="nearest")

    train_data = train_datagen.flow_from_directory(train_dir,
                                              target_size=(150,150),
                                              batch_size=32,
                                              class_mode="categorical")



    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_data = test_datagen.flow_from_directory(test_dir,
                                            target_size=(150,150),
                                            batch_size=32,
                                            class_mode="categorical")

#=================================================================
    chanDim = 1
    model = Sequential(name="SoilNet")
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=(150,150,3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (4, 4), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))

    model.add(Conv2D(128, (4, 4), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation("softmax"))


    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    reduction_lr = ReduceLROnPlateau(monitor = "val_accuracy",patience = 2 ,verbose = 1, factor = 0.2, min_lr = 0.00001)
    model.summary()
    # plot_model(model,show_shapes=True)

    callback_list = [reduction_lr]

    start = time.time()

    history = model.fit_generator(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=callback_list  # Make sure this is correctly defined
    )

    end = time.time()
    print("Total train time: ", (end - start) / 60, " mins")

    def plot_graph(history,string):
            plt.figure(figsize=(12,8))
            plt.plot(history.history[string],label=str(string))
            plt.plot(history.history["val_"+str(string)],label="val_"+str(string))
            plt.xlabel("Epochs")
            plt.ylabel(str(string))
            plt.legend()
            plt.show()
    plot_graph(history,"accuracy")
    plot_graph(history,"loss")

    model.save("SoilNet.h5")

    from IPython.display import FileLink
    FileLink('SoilNet.h5')
    acc = history.history['accuracy'][-1]
    loss = history.history['accuracy'][-1]

    return render(request,"users/training.html",{"Accuracy":acc,"Loss":loss})


def prediction(request):
    from django.conf import settings
    import os
    if request.method=='POST':
        from django.core.files.storage import FileSystemStorage
        from tensorflow.keras.models import load_model
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/")
        filename = fs.save(image_file.name, image_file)
        # detect_filename = fs.save(image_file.name, image_file)
        uploaded_file_url = "/media/Black Soil/" + filename  # fs.url(filename)
        print("Image path ", uploaded_file_url)
        model_path =os.path.join(settings.MEDIA_ROOT, 'SoilNet.h5')
        file=os.path.join(settings.MEDIA_ROOT, filename) 
        SoilNet = load_model(model_path)
        from .utility.Algorithm import model_predict
        prediction,output_page = model_predict(file,SoilNet)
        print("Result=", prediction)
        return render(request, "users/testform.html", {'path': prediction,'result': output_page})
    else:
        return render(request, "users/testform.html",{})

