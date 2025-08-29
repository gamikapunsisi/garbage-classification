from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class GarbageClassifier:
    def __init__(self, input_shape=(128, 128, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def summary(self):
        return self.model.summary()