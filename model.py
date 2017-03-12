import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.layers import Input, Flatten, Dense, Convolution2D, Lambda, Cropping2D, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils.visualize_util import plot
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv as csv
import cv2
import numpy as np
import sklearn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('load_model',True,'Load existing model bool')
flags.DEFINE_bool('debug_set_train', False, 'Debug set train on 3 images')
flags.DEFINE_bool('debug_set_test', False, 'Debug set test on 3 images')
flags.DEFINE_string('driving_log','./driving_log.csv', 'Driving log csv file')
flags.DEFINE_string('img_dir', './IMG/', 'Image directory')
flags.DEFINE_string('model_name', 'model.h5', 'Model output path')
flags.DEFINE_float('validation_split', 0.2, 'Validation split')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epochs', 5, 'Number of Epochs')

# command line flags

def load_driving_log_csv(log_file='driving_log.csv', date=None):
    """
    Utility function to load driving log csv file

    Arguments:
        log_file - String
        date - Date object or none for all dates
    """
    print("Driving log file", log_file)

    lines = []

    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

def normalize_image(X_train):
    return (X_train - 127.5) / 255.

def load_image(path,image_dir):
    filename = path.split('/')[-1]
    current_path = image_dir + filename
    image = cv2.imread(current_path)
    return cv2.resize(image,None,None,fx=0.5,fy=0.5)


def load_data(csv_line, images, angles, image_dir=FLAGS.img_dir, include_flipped=True, include_left_right=True):
    center_image = load_image(csv_line[0],image_dir)
    if center_image is None:
        print("None image")
    center_angle = float(csv_line[3])
    images.append(center_image)
    angles.append(center_angle)

    if include_flipped is True:
        image_flipped = np.fliplr(center_image)
        measurement_flipped = -center_angle
        images.append(image_flipped)
        angles.append(measurement_flipped)

    if include_left_right is True:
        # create adjusted steering measurements for the side camera images
        correction = 0.2  # this is a parameter to tune
        steering_left = center_angle + correction
        steering_right = center_angle - correction

        left_image = load_image(csv_line[1],image_dir)
        right_image = load_image(csv_line[2],image_dir)
        images.append(left_image)
        angles.append(steering_left)
        images.append(right_image)
        angles.append(steering_right)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                load_data(batch_sample,images,angles)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def main(_):
    # load bottleneck data
    samples = []
    X_train = []
    y_train = []
    driving_log = FLAGS.driving_log
    validation_split = FLAGS.validation_split
    if FLAGS.debug_set_train is True or FLAGS.debug_set_test is True:
        driving_log = './driving_log_dev.csv'

    samples = load_driving_log_csv(log_file=driving_log)

    if FLAGS.debug_set_train is False and FLAGS.debug_set_test is False:
        train_samples, validation_samples = train_test_split(samples, test_size=validation_split)
        # compile and train the model using the generator function
        train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
        validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)
    else:
        images = []
        angles = []
        for line in samples:
            load_data(line,images,angles,include_flipped=FLAGS.debug_set_train)
        X_train = np.array(images)[0:3,:,:,:]
        y_train = np.array(angles)[0:3]
        X_validation = np.array(images)
        Y_validation = np.array(angles)

    width,height = 320,160
    scaled_width,scaled_height = width*0.5,height*0.5
    top_crop, bot_crop = int(0.34*scaled_height), int(0.15*scaled_height)
    ch, row, col = 3, scaled_height - (top_crop+bot_crop), scaled_width  # Trimmed image format

    #NVIDIA Architecture
    if FLAGS.load_model is False:
        model = Sequential()
        # Preprocess incoming data, centered around zero with small standard deviÎation
        model.add(Cropping2D(cropping=((top_crop, bot_crop), (0, 0)), input_shape=(int(scaled_height), int(scaled_width), 3)))
        model.add(Lambda(lambda x: x / 255. - 0.5,
                         input_shape=(row, col, ch),
                         output_shape=(row, col, ch)))
        model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
        model.add(Convolution2D(36, 3, 3, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(48, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(100,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

    else:
        model = load_model(FLAGS.model_name)

    adam = Adam(lr=FLAGS.learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    tbCallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    plot(model, to_file='model.png', show_shapes=True)

    if FLAGS.debug_set_test is True:
        predict = model.predict(X_train,verbose=1)
        print(predict)
    elif FLAGS.debug_set_train is False:
        history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples*4), validation_data = validation_generator,
                        nb_val_samples = len(validation_samples), nb_epoch = FLAGS.epochs, verbose=1, callbacks=[tbCallback])
        print(history_object.history.keys())

        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig('train_history.jpg')
    else:
        model.fit(X_train,y_train,batch_size=FLAGS.batch_size,nb_epoch=FLAGS.epochs)
        e = model.evaluate(X_train,y_train)
        print(e)
    model.save(FLAGS.model_name)

    import gc;
    gc.collect()

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

