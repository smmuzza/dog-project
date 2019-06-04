from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('data/dog_images/train')
valid_files, valid_targets = load_dataset('data/dog_images/valid')
test_files, test_targets = load_dataset('data/dog_images/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("/data/dog_images/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("data/lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))




import cv2                
import matplotlib.pyplot as plt                        
#%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()




# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0




human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
humansDetectedAsHumans = 0
for i in human_files_short:
    success = face_detector(i)
    #print('human face_detector access success:', success)
    if success:
        humansDetectedAsHumans += 1

dogsDetectedAsHumans = 0
for i in dog_files_short:
    success = face_detector(i)
    #print('dog face_detector access success:', success)
    if success:
        dogsDetectedAsHumans += 1     
    
print(humansDetectedAsHumans / human_files_short.shape[0])
print(dogsDetectedAsHumans / dog_files_short.shape[0])  



## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.

# extract pre-trained face detector lbpcascade
lbp_face_cascade = cv2.CascadeClassifier('lbpcascade/lbpcascade_frontalface_improved.xml')  

# load color (BGR) image
img = cv2.imread(human_files[3])
#img = cv2.imread('Aaron_Eckhart_0001.jpg')
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = lbp_face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = lbp_face_cascade.detectMultiScale(gray)
    return len(faces) > 0

humansDetectedAsHumans = 0
for i in human_files_short:
    success = face_detector(i)
    #print('human face_detector access success:', success)
    if success:
        humansDetectedAsHumans += 1

dogsDetectedAsHumans = 0
for i in dog_files_short:
    success = face_detector(i)
    #print('dog face_detector access success:', success)
    if success:
        dogsDetectedAsHumans += 1     
    
print(humansDetectedAsHumans / human_files_short.shape[0])
print(dogsDetectedAsHumans / dog_files_short.shape[0])  



from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')



from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))




### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 




### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm 
## on the images in human_files_short and dog_files_short.
humansDetectedAsHumans = 0
for i in human_files_short:
    success = dog_detector(i)
    print('human face_detector access success:', success)
    if success:
        humansDetectedAsHumans += 1

dogsDetectedAsHumans = 0
for i in dog_files_short:
    success = dog_detector(i)
    print('dog face_detector access success:', success)
    if success:
        dogsDetectedAsHumans += 1     
    
print(humansDetectedAsHumans / human_files_short.shape[0])
print(dogsDetectedAsHumans / dog_files_short.shape[0])  



from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255



from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

### TODO: Define your architecture.
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(133, activation='softmax'))

model.summary()

# best Test accuracy: 8.3732% with cifar10_cnn network, 5 epochs
# best with additional cov layer is Test accuracy: > 10.0%, 10 epochs





model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)



model.load_weights('saved_models/weights.best.from_scratch.hdf5')



# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)



bottleneck_features = np.load('data/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']


VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()


VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5', 
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)



VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')


# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)



from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]



### TODO: Obtain bottleneck features from another pre-trained CNN.
useVGG19 = 1
useResnet50 = 1
useDataGenResnet50 = 0

if useVGG19:
    bottleneck_features = np.load('data/bottleneck_features/DogVGG19Data.npz')
    train_VGG19 = bottleneck_features['train']
    valid_VGG19 = bottleneck_features['valid']
    test_VGG19 = bottleneck_features['test']
    
if useResnet50:
    bottleneck_features = np.load('data/bottleneck_features/DogResnet50Data.npz')
    train_Resnet50 = bottleneck_features['train']
    valid_Resnet50 = bottleneck_features['valid']
    test_Resnet50 = bottleneck_features['test']    
    
    
    
    ### TODO: Define your architecture.

from keras.layers import Dense, Flatten, MaxPooling1D
from keras.models import Sequential

dropout = 0.5
acti = 'tanh' # tried sigmoid, tanh (best), relu, elu

if useVGG19:
    print('VGG + Head Arch')
    VGG19_CustomHead = Sequential()
    VGG19_CustomHead.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
    VGG19_CustomHead.add(Dense(512, activation=acti)) 
    VGG19_CustomHead.add(Dropout(dropout))
    VGG19_CustomHead.add(Dense(133, activation='softmax'))
    VGG19_CustomHead.summary()
    
if useResnet50:
    print('Resnet50 + Head Arch')
    Resnet50_CustomHead = Sequential()
    Resnet50_CustomHead.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
    Resnet50_CustomHead.add(Dense(512, activation=acti)) # sigmoid, tanh, relu
    Resnet50_CustomHead.add(Dropout(dropout))
    Resnet50_CustomHead.add(Dense(133, activation='softmax'))
    Resnet50_CustomHead.summary()    
    
 
### TODO: Compile the model.
if useVGG19:
    VGG19_CustomHead.compile(loss='categorical_crossentropy', 
                             optimizer='rmsprop', 
                             metrics=['accuracy'])
    
if useResnet50:
    Resnet50_CustomHead.compile(loss='categorical_crossentropy', 
                             optimizer='rmsprop', 
                             metrics=['accuracy'])  
    
    
from keras.preprocessing.image import ImageDataGenerator

if useDataGenResnet50: 
    ### Data Augmentation

    # create and configure augmented image generator
    datagen_train = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
        horizontal_flip=True) # randomly flip images horizontally

    # print shape of training set
    print('train_tensors shape:', train_tensors.shape)

    # fit augmented image generator on data
    datagen_train.fit(train_tensors)

    # create and configure augmented image generator
    datagen_valid = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
        horizontal_flip=True) # randomly flip images horizontally

    # print shape of training set
    print('valid_tensors shape:', valid_tensors.shape)

    # fit augmented image generator on data
    datagen_train.fit(valid_tensors)    



### TODO: Train the model.
from keras.callbacks import ModelCheckpoint, EarlyStopping

# train the model
batch_size = 32 # 8, 16, 32, train_Resnet50.shape[0], seem to have enough memory to handle this
epochs = 20

if useVGG19:
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG19.hdf5', 
                               verbose=1, save_best_only=True)
    VGG19_CustomHead.fit(train_VGG19, train_targets, 
                     epochs=20, batch_size=batch_size, 
                     validation_data=(valid_VGG19, valid_targets), 
                     callbacks=[checkpointer], verbose=1, shuffle=True)

if useResnet50:
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5', 
                               verbose=1, save_best_only=True)
    #EarlyStopping(monitor='val_loss', 
    #              min_delta=0, patience=3, 
    #              verbose=0, mode='auto', 
    #              baseline=None, restore_best_weights=False)
    
    if useDataGenResnet50:               
        Resnet50_CustomHead.fit_generator(datagen_train.flow(train_Resnet50, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_Resnet50.shape[0] // batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpointer],
                    validation_data=datagen_valid.flow(valid_Resnet50, valid_targets, batch_size=batch_size),
                    validation_steps=valid_Resnet50.shape[0] // batch_size)
    else:
        Resnet50_CustomHead.fit(train_Resnet50, train_targets, 
                                epochs=epochs, batch_size=batch_size,
                                validation_data=(valid_Resnet50, valid_targets), 
                                callbacks=[checkpointer], 
                                verbose=1, shuffle=True)    
    
    

### TODO: Load the model weights with the best validation loss.
if useVGG19:
    VGG19_CustomHead.load_weights('saved_models/weights.best.VGG19.hdf5')
    
if useResnet50:
    Resnet50_CustomHead.load_weights('saved_models/weights.best.Resnet50.hdf5')    


    ### TODO: Calculate classification accuracy on the test dataset.
if useVGG19:
    # get index of predicted dog breed for each image in test set
    VGG19_predictions = [np.argmax(VGG19_CustomHead.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
    print('VGG19 Test Accuracy: %.4f%%' % test_accuracy)  
    # best to date 79.0-80.0%, 2x 500 layers, dropout, tanh activation
    
if useResnet50:
    # get index of predicted dog breed for each image in test set
    Resnet50_predictions = [np.argmax(Resnet50_CustomHead.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
    print('Resnet50 Test Accuracy: %.4f%%' % test_accuracy)    
    # best to date  84.2105%, no data augmentation, 500 dense hidden layer, 0.5 dropout, tanh
    # best to date with elu  79.0670% (but trains fast), no data augmentation, 500 layer, 0.5 dropout
    # best to date with data augmentation and elu 77%
    
    
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

from extract_bottleneck_features import *

if useVGG19:
    def VGG19_predict_breed(img_path):
        # extract bottleneck features
        bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = VGG19_CustomHead.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        predIdx = np.argmax(predicted_vector)
        print(predIdx)
        return dog_names[predIdx]

img_path = human_files[0]
showImg(human_files[0])
bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
# obtain predicted vector
predicted_vector = VGG19_CustomHead.predict(bottleneck_feature)
# return dog breed that is predicted by the model
predIdx = np.argmax(predicted_vector)
print(predIdx)
print(dog_names[0])
print(dog_names[predIdx])


#result = VGG19_predict_breed(human_files[0])
#print(result)    
    
if useResnet50:
    def Resnet50_predict_breed(img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = Resnet50_CustomHead.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        predIdx = np.argmax(predicted_vector)
        return dog_names[predIdx]    


    


