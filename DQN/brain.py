
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras import backend as K

import tensorflow as tf

        
def huber_loss(y_true, y_pred):
    HUBER_LOSS_DELTA = 1.0
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

class Brain:
    LEARNING_RATE = 0.00025

    def __init__(self, stateDataCount, actionCount):
        self.stateDataCount = stateDataCount
        self.actionCount = actionCount

        self.model = self._createModel()        # online network
        self.target_model = self._createModel() # target network

    def _createModel(self):
        model = Sequential()

        more_deep_setting = True
        print("More deep setting", more_deep_setting)
        deep_setting = False
        print("Deep setting", deep_setting)

        if more_deep_setting:
            model.add(Dense(units=32, activation='relu', 
                            input_dim=self.stateDataCount))
            model.add(Dense(units=16, activation='relu'))
            model.add(Dense(units=8, activation='relu'))

        else: # Normal layer (breath)
            model.add(Dense(units=64, activation='relu', 
                            input_dim=self.stateDataCount))
            
            if deep_setting:    # add one more hidden layer
                model.add(Dense(units=32, activation='relu'))
            
        model.add(Dense(units=self.actionCount, activation='linear'))
        
        # Learning method
        optimizer = RMSprop(lr=Brain.LEARNING_RATE)

        # Set loss function
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def train(self, x, y, epoch=1):
        history_callback = self.model.fit(x, y, batch_size=64, epochs=epoch, 
                                        verbose=0)
        return history_callback.history["loss"]


    def predict(self, state):
        return self.model.predict(state)

    def predictOne(self, state):
        return self.predict(state.reshape(1, self.stateDataCount)).flatten()
    
    def predict_target(self, state):
        return self.target_model.predict(state)

    def predictOne_target(self, state):
        return self.predict_target(state.reshape(1, self.stateDataCount)).flatten()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    