from models import RNN, LSTM, GRU
from utils.data_loader import TimeSeriesDataLoader
from utils.model_manager import ModelManager


# Configuration
file_path = 'air-quality-forecasting/data/raw/ankhanh_measurements.csv'

features_type='M'
input_size = 72
label_size = 24
offset = 24
train_size = 0.70
val_size = 0.15
batch_size = 32

num_epochs = 50
patience = 20
learning_rate = 0.001
hidden_size = 32
num_layers = 2



data_loader = TimeSeriesDataLoader(file_path,
                                    input_size=input_size,
                                    label_size=label_size,
                                    offset=offset,
                                    train_size=train_size,
                                    val_size=val_size,
                                    features_type=features_type,
                                    batch_size=batch_size)

# Khởi tạo mô hình RNN
RNN_model = RNN(
    input_size=data_loader.in_variable,
    hidden_size=hidden_size,
    output_size=data_loader.out_variable,
    ahead=label_size,
    num_layers=num_layers,
)

# Khởi tạo quản lý mô hình
RNN_manager = ModelManager(
    model=RNN_model,
    train_loader=data_loader.train_loader,
    val_loader=data_loader.val_loader,
    lr=learning_rate,
    patience=patience
)

# Huấn luyện mô hình
RNN_manager.train(
    num_epochs=num_epochs,
    save_dir='air-quality-forecasting/model'
)