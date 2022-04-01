
import keras as K

class DQNetwork:

    def __init__(self, numStates, numActions, maxAction=1.0, minAction=0.0, layerSizes=(64, 64),
                 batchNormPerLayer=(True, True), dropoutPerLayer=(0, 0), learningRate=0.0001):
        self.numStates = numStates
        self.numActions = numActions
        self.maxAction = maxAction
        self.minAction = minAction
        self.layerSizes = layerSizes
        self.batchNormPerLayer = batchNormPerLayer
        self.dropoutPerLayer = dropoutPerLayer
        self.learningRate = learningRate

        self.build_model()

    def build_model(self):
        states = K.layers.Input(shape=(self.numStates,), name='states')
        neuralNet = states
        # hidden layers

        for i in range(len(self.layerSizes)):
            neuralNet = K.layers.Dense(units=self.layerSizes[i])(neuralNet)
            neuralNet = K.layers.Activation('relu')(neuralNet)
            if self.batchNormPerLayer[i]:
                neuralNet = K.layers.BatchNormalization()(neuralNet)
            neuralNet = K.layers.Dropout(self.dropoutPerLayer[i])(neuralNet)

        actions = K.layers.Dense(units=self.numActions, activation='linear',
                                 name='rawActions')(neuralNet)

        self.model = K.models.Model(inputs=states, outputs=actions)

        self.optimizer = K.optimizers.Adam(lr=self.learningRate)
        self.model.compile(loss='mse', optimizer=self.optimizer)