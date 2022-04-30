import keras 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D


def simple_model(input_shape,no_classes):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2)))
	
	model.add(Flatten())
	
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(no_classes, activation='softmax'))
	
	model.summary()
	return(model)
	
def alexnet(input_shape,no_classes):
	print('Inializing AlexNet Model....!')
	#Instantiate an empty model
	model = Sequential()
	
	# 1st Convolutional Layer
	model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	
	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	
	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	
	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	
	# Passing it to a Fully Connected layer
	model.add(Flatten())
	
	
	# 1st Fully Connected Layer
	model.add(Dense(4096, input_shape=(224*224*3,)))
	model.add(Activation('relu'))
	# Add Dropout to prevent overfitting
	model.add(Dropout(0.4))
	
	# 2nd Fully Connected Layer
	model.add(Dense(4096))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))
	
	# 3rd Fully Connected Layer
	model.add(Dense(1000))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))
	
	# Output Layer
	model.add(Dense(no_classes))#17 are the number of output(classes)
	model.add(Activation('softmax'))
	
	model.summary()
	
	return(model)


def VGG_16(input_shape,no_classes):
    print('Inializing VGG16 Model....!')
    
    base_model = keras.applications.vgg16.VGG16(input_shape=input_shape,include_top=False, weights='imagenet')
    #fine Tuning of model
    """
    base_model.trainable = True
    set_trainable=False
    
    for layer in base_model.layers:
        if layer.name=='block5_conv1':
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False
    """
    #### adding extra layers to pre-trained model
    
    model = keras.Sequential([base_model,
						keras.layers.GlobalAveragePooling2D(),
						keras.layers.Dense(1024, 
						activation='relu'),
						keras.layers.Dropout(0.3),
						keras.layers.Dense(1024,activation='relu'),
						keras.layers.Dropout(0.3),
						keras.layers.Dense(512,activation='relu'),
						keras.layers.Dense(no_classes, activation='softmax')])
    model.summary()
    return(model)
   
    
def VGG_19(input_shape,no_classes):
	print('Inializing VGG19 Model....!')
	base_model = keras.applications.vgg19.VGG19(input_shape=input_shape,
													include_top=False, weights='imagenet')
	

	
	base_model.trainable = True
	set_trainable=False
	
	for layer in base_model.layers:
		if layer.name=='block5_conv1':
			set_trainable=True
		if set_trainable:
			layer.trainable=True
		else:
			layer.trainable=False
	#### adding extra layers to pre-trained model
	
	model = keras.Sequential([base_model,
						keras.layers.GlobalAveragePooling2D(),
						keras.layers.Dense(1024, 
						activation='relu'),
						keras.layers.Dropout(0.3),
						keras.layers.Dense(1024,activation='relu'),
						keras.layers.Dropout(0.3),
                        keras.layers.BatchNormalization(),
						keras.layers.Dense(512,activation='relu'),
						keras.layers.Dense(no_classes, activation='softmax')])
	
	return(model)

def inception_v3(input_shape,no_classes):
	print('Inializing INCEPTION V3 Model....!')
	base_model = keras.applications.inception_v3.InceptionV3(input_shape=input_shape,
													include_top=False, weights='imagenet')

	for layer in base_model.layers:
		layer.trainable=False
	#### adding extra layers to pre-trained model
	
	model = keras.Sequential([base_model,
						keras.layers.GlobalAveragePooling2D(),
						keras.layers.Dense(1024, 
						activation='relu'),
						keras.layers.Dropout(0.2),
						keras.layers.Dense(1024,activation='relu'),
						keras.layers.Dropout(0.2),
						keras.layers.Dense(512,activation='relu'),
						keras.layers.Dense(no_classes, activation='softmax')])
	
	return(model)

def EfficientNetB0(input_shape,no_classes):
    import efficientnet.keras as efn 
    base_model = efn.EfficientNetB0(include_top=False,weights='imagenet',input_shape=input_shape)
   # for layer in base_model.layers:
    #    layer.trainable=True
        #### adding extra layers to pre-trained model
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dense(1024, activation='relu'),
                              keras.layers.BatchNormalization(),
                              keras.layers.Dropout(0.5),
                              #keras.layers.Dense(1024,activation='relu'),
                             #keras.layers.Dense(512,activation='relu'),
                             # keras.layers.Dropout(0.5),
                              keras.layers.Dense(no_classes, activation='softmax')])
    #base_model.summary()
    model.summary()
    return(model)
    

def ResNet_101(input_shape,no_classes):
	from keras.applications import ResNet101
	print('Inializing ResNet101 Model....!')
	base_model = tf.keras.applications.ResNet101(input_shape=input_shape,
													include_top=False, weights='imagenet')

	for layer in base_model.layers:
		layer.trainable=False
	#### adding extra layers to pre-trained model
	
	model = keras.Sequential([base_model,
						keras.layers.GlobalAveragePooling2D(),
						keras.layers.Dense(1024, 
						activation='relu'),
						keras.layers.Dropout(0.2),
						keras.layers.Dense(1024,activation='relu'),
						keras.layers.Dropout(0.2),
						keras.layers.Dense(512,activation='relu'),
						keras.layers.Dense(no_classes, activation='softmax')])
	
	return(model)

def ResNet_152(input_shape,no_classes):
	print('Inializing ResNet152 Model....!')
	base_model = keras.applications.ResNet152.Resnet152(input_shape=input_shape,
													include_top=False, weights='imagenet')

	for layer in base_model.layers:
		layer.trainable=False
	#### adding extra layers to pre-trained model
	
	model = keras.Sequential([base_model,
						keras.layers.GlobalAveragePooling2D(),
						keras.layers.Dense(1024, 
						activation='relu'),
						keras.layers.Dropout(0.2),
						keras.layers.Dense(1024,activation='relu'),
						keras.layers.Dropout(0.2),
						keras.layers.Dense(512,activation='relu'),
						keras.layers.Dense(no_classes, activation='softmax')])
	
	return(model)

