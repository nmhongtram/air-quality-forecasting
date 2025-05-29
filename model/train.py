from model.models import RNN, LSTM, GRU
from model.utils.data_loader import TimeSeriesDataLoader
from model.utils.model_manager import ModelManager
from model.config import Configuration


# Load cấu hình
c = Configuration()
file_path = 'data/raw/ankhanh_measurements.csv'

# Load dữ liệu
data_loader = TimeSeriesDataLoader(
    file_path=file_path,
    input_size=c.DATA_INPUT_SIZE,
    label_size=c.DATA_LABEL_SIZE,
    offset=c.DATA_OFFSET,
    train_size=c.DATA_TRAIN_SIZE,
    val_size=c.DATA_VAL_SIZE,
    features_type=c.FEATURES_TYPE,
    batch_size=c.BATCH_SIZE,
    stride=c.STRIDE
)

# Các mô hình cần huấn luyện
model_classes = {
    "RNN": RNN,
    "LSTM": LSTM,
    "GRU": GRU
}

# Huấn luyện từng mô hình
for model_name, model_class in model_classes.items():
    print(f"\n===== Training {model_name} =====")

    print(f"Input size: {data_loader.in_variable}")
    print(f"Output size: {data_loader.out_variable}")
    
    
    model = model_class(
        input_size=data_loader.in_variable,
        hidden_size=c.MODEL_HIDDEN_SIZE,
        output_size=data_loader.out_variable,
        ahead=c.MODEL_AHEAD,
        num_layers=c.MODEL_NUM_LAYERS
    )

    manager = ModelManager(
        model=model,
        train_loader=data_loader.train_loader,
        val_loader=data_loader.val_loader,
        lr=c.LEARNING_RATE,
        patience=c.PATIENCE
    )

    manager.train(
        num_epochs=c.NUM_EPOCHS,
        save_dir='model'
    )

print("\nAll models trained and saved.")
