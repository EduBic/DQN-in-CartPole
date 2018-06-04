

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class Brain:

    def __init__(self, stateDataCount, actionCount):
        self.stateDataCount = stateDataCount
        self.actionCount = actionCount

        self.model = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', 
            input_dim=self.stateDataCount))
        model.add(Dense(units=self.actionCount, activation='linear'))
        
        # Learning method
        optimizer = RMSprop(lr=0.00025)

        # Set loss function
        model.compile(loss='mse', optimizer=optimizer)

        return model

    def train(self, x, y, epoch=1):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def predictOne(self, state):
        return self.predict(state.reshape(1, self.stateDataCount)).flatten()