from model.models import RNN, LSTM, GRU
from utils.data_loader import TimeSeriesDataLoader
from utils.model_manager import ModelManager
from model.config import Configuration


# Configuration
file_path = 'data/raw/ankhanh_measurements.csv'

# features_type='M'
# input_size = 72
# label_size = 24
# offset = 24
# train_size = 0.70
# val_size = 0.15
# batch_size = 32
# stride = 6

# num_epochs = 50
# patience = 20
# learning_rate = 0.001
# hidden_size = 32
# num_layers = 2

c = Configuration()

data_loader = TimeSeriesDataLoader(file_path,
                                    input_size=c.DATA_INPUT_SIZE,
                                    label_size=c.DATA_LABEL_SIZE,
                                    offset=c.DATA_OFFSET,
                                    train_size=c.DATA_TRAIN_SIZE,
                                    val_size=c.DATA_VAL_SIZE,
                                    features_type=c.FEATURES_TYPE,
                                    batch_size=c.BATCH_SIZE, 
                                    stride=c.STRIDE)

# Khởi tạo mô hình LSTM
LSTM_model = LSTM(
    input_size=data_loader.in_variable,
    hidden_size=c.MODEL_HIDDEN_SIZE,
    output_size=data_loader.out_variable,
    ahead=c.MODEL_AHEAD,
    num_layers=c.MODEL_NUM_LAYERS,
)

# Khởi tạo quản lý mô hình
LSTM_manager = ModelManager(
    model=LSTM_model,
    train_loader=data_loader.train_loader,
    val_loader=data_loader.val_loader,
    lr=c.LEARNING_RATE,
    patience=c.PATIENCE
)

# Huấn luyện mô hình
LSTM_manager.train(
    num_epochs=c.NUM_EPOCHS,
    save_dir='model'
)