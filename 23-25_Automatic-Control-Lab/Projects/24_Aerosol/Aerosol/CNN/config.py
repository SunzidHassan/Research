config = {
    'seed': 42,
    'X_train': './Aerosol/data/X_train.csv',
    'y_train': './Aerosol/data/y_train.csv',
    'X_test': './Aerosol/data/X_test.csv',
    'y_test': './Aerosol/data/y_test.csv',
    'n_epochs': 4,
    'batch_size': 128,
    'learning_rate': 1e-4,
    'dropout': 0.3,
    'weight_decay': 1e-2,
    'early_stop': 2,
    'hidden_layers': 3,
    'hidden_units': 128,
    'kernel_size': 3,
    'filters': 128,
    'save_path': './models/dec22.pth',  # Your model will be saved here.
    'window_size': 4,
    'label_size': 1,
    'shift': 1
    }   