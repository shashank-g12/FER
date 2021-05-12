import os
import cv2
import glob
import numpy as np

def preprocessing(dir):
	
	data_dir = os.getcwd()
	path = 'dataset/' + dir
	tdata_dir = os.path.join(data_dir, path)

	labels = ['neutral', 'happy', 'surprise', 'angry', 'sad', 'fear', 'disgust']
	emotion = { 'neutral':0,
				'happy':1,
				'surprise':2,
				'angry':3,
				'sad':4,
				'fear':5,
				'disgust':6 }

	x_train = []
	y_train = []
	count = 0
	for i in labels:
		image_list = list(glob.glob(tdata_dir + '/'+ i +'/*.jpg'))
		print('Reading '+ i + ' folder...')
		for j in image_list:
			img = cv2.imread(j)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			x_train.append(gray)
			y_train.append(emotion[i])
		print('Read '+ str(len(image_list)) + ' files')
		count += len(image_list)
	print('------------------------------------------')
	print('Read '+ count + ' files from ' + dir + 'ing dataset')
	print('------------------------------------------')
	print()
	return x_train,y_train


def shuffle_data(a,b):
    c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)]
    np.random.shuffle(c)
    a2 = c[:, :a.size//len(a)].reshape(a.shape)
    b2 = c[:, a.size//len(a):].reshape(b.shape)
    return a2,b2