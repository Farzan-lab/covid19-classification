
"""
## Setup
"""

import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalMaxPooling2D, Conv1D, Flatten
from keras.applications.mobilenet_v2 import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from data import get_train_data
from sklearn.utils import class_weight



"""
## Prepare the data
"""

data, labels = get_train_data(pre_process=True)
X_train, X_test, Y_train, Y_test = train_test_split(data, 
                                                    labels,
                                                    stratify=labels,
                                                    random_state=42,
                                                    shuffle=True)

encode = LabelEncoder()
onehotencoder = OneHotEncoder()

Y_train = encode.fit_transform(Y_train)
train_labels = Y_train.reshape(-1,1)

Y_test = encode.transform(Y_test)
test_labels = Y_test.reshape(-1,1)

Y_train = onehotencoder.fit_transform(train_labels)
Y_train = Y_train.toarray()

Y_test = onehotencoder.transform(test_labels)
Y_test = Y_test.toarray()

"""
## Build the model
"""
mobile = MobileNetV2(input_shape= (112,112,3),  include_top=False , weights="imagenet", pooling='max')
mobile.trainable = False
model = Sequential()

model.add(Conv1D(3, 1, activation='relu', padding='same', input_shape=(112, 112, 1)))
model.add(mobile)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='sigmoid'))
model.summary()

"""
## Train the model
"""

batch_size = 512
epochs = 20

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(test_labels),
                                                 test_labels[:,0])

class_weights = {i: class_weights[i] for i in range(0, len(class_weights))}

model.fit(X_train, Y_train, batch_size=batch_size , epochs=epochs , validation_data = (X_test, Y_test), class_weight= class_weights )

"""
## Evaluate the trained model
"""

score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save_weights('weights/mobilenet.hdf5')
