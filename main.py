import preprocess
import face_detection
import model
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from confusion_matrix import plot_confusion_matrix

n_classes = 7
x_train, y_train = preprocess.preprocessing('train')
x_test, y_test = preprocess.preprocessing('test')

x_train = face_detection.face_detect(x_train)                   # face detection
x_test = face_detection.face_detect(x_test)               


X_train = np.asarray(x_train, dtype = 'float32')
X_test = np.asarray(x_test, dtype = 'float32')
Y_train = np.asarray(y_train)
Y_test = np.asarray(y_test)

X_train = X_train.reshape((len(X_train),48, 48, 1))
X_test = X_test.reshape((len(X_test), 48, 48, 1))

print('Train data:{}'.format(X_train.shape))
print('Train classes:{}'.format(Y_train.shape))
print('Test data:{}'.format(X_test.shape))
print('Test classes:{}'.format(Y_test.shape))

X_train, Y_train = preprocess.shuffle_data(X_train, Y_train)  # shuffle data
X_train, Y_train = preprocess.shuffle_data(X_train, Y_train)  

X_train = X_train/255.                                        # normalize data
X_test = X_test/255.

label = { 0:'neutral',1:'happy',2:'surprise',3:'angry',4:'sad',5:'fear',6:'disgust'}
classes = ['neutral', 'happy', 'surprise', 'angry', 'sad', 'fear', 'disgust']

print('Some images from the training dataset')
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Truth: {}".format(label[Y_train[i]]))
    plt.xticks([])
    plt.yticks([])
plt.show()

Y_train = np_utils.to_categorical(Y_train, n_classes)   # one hot encoding
Y_test = np_utils.to_categorical(Y_test, n_classes)

model = model.build_model()
print('Model Summary')
print(model.summary())

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

checkpoint = ModelCheckpoint("fer_model.h5", monitor='val_accuracy',\
	 verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')

print('Training model...')
history = model.fit(X_train, Y_train,batch_size=64, epochs=20,verbose=1,shuffle=True,\
	validation_data=(X_test, Y_test),validation_batch_size = 128,callbacks = [checkpoint, early])

fig = plt.figure()
plt.subplot(1,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

fig = plt.figure()
plt.subplot(1,1,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

print('Loading trained model...')
correct_model = load_model('fer_model.h5')
YY_pred = correct_model.predict(X_test, batch_size = 64, verbose = 1)

Y_prediction = np.argmax(YY_pred, axis=1)
Y_true = np.argmax(Y_test, axis = 1)

cm = confusion_matrix(Y_true, Y_prediction)
plt = plot_confusion_matrix(cm = cm, target_names = classes , title = 'Confusion Matrix', normalize = True)
plt.show()

print('Some other metrics')
print(classification_report(Y_true, Y_prediction, target_names = classes))

correct_indices = np.nonzero(Y_prediction == Y_true)[0]
incorrect_indices = np.nonzero(Y_prediction != Y_true)[0]
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")

plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(X_test[correct], cmap='gray', interpolation='none')
    plt.title(
      "P: {},T: {}".format(classes[Y_prediction[correct]],
                                        classes[Y_true[correct]]))
    plt.xticks([])
    plt.yticks([])
plt.subplots_adjust(wspace = 0.5, hspace= 0.5)
plt.show()
    
figure_evaluation = plt.figure()
# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[10:19]):
    plt.subplot(6,3,i+10)
    plt.imshow(X_test[incorrect], cmap='gray', interpolation='none')
    plt.title(
      "P: {},T:{}".format(classes[Y_prediction[incorrect]], 
                                       classes[Y_true[incorrect]]))
    plt.xticks([])
    plt.yticks([])

plt.subplots_adjust(wspace = 0.5, hspace= 0.5)
plt.show()