class Configuration:
    def __init__(self):
        # Model structure
        self.MODEL_INPUT_SIZE = 8
        self.MODEL_HIDDEN_SIZE = 32
        self.MODEL_OUTPUT_SIZE = 3
        self.MODEL_AHEAD = 24
        self.MODEL_NUM_LAYERS = 2

        # Dataset preparation
        self.FEATURES_TYPE = 'M'  # Multi-feature
        self.DATA_INPUT_SIZE = 72
        self.DATA_LABEL_SIZE = 24
        self.DATA_OFFSET = 24
        self.DATA_TRAIN_SIZE = 0.70
        self.DATA_VAL_SIZE = 0.15
        self.STRIDE = 6
        self.BATCH_SIZE = 32

        # Training parameters
        self.NUM_EPOCHS = 50
        self.PATIENCE = 20
        self.LEARNING_RATE = 0.001
        self.HIDDEN_SIZE = 32
        self.NUM_LAYERS = 2

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        return '\n'.join([f"{k}: {v}" for k, v in self.__dict__.items()])
