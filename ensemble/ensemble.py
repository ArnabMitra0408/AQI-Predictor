import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import json
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()

# Load environment variables
final_data_path = os.getenv('final_data_path')
rf_model_path = os.getenv('rf_model_path_ensemble')
lstm_model_path = os.getenv('lstm_model_path_ensemble')
lstm_loss_path = os.getenv('lstm_loss_path_ensemble')
feature_importance_path = os.getenv('feature_importance_path_ensemble')
model_metrics_path = os.getenv('model_metrics_path_ensemble')
test_predictions_path = os.getenv('test_predictions_path_ensemble')
test_size = float(os.getenv('test_size'))

class TimeSeriesDataset(Dataset):
    def __init__(self, features, targets, sequence_length=24):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets.values.reshape(-1, 1))
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
        
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return X, y

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return out

def train_lstm(train_loader, val_loader, model, epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), lstm_model_path)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def make_ensemble_predictions(X, rf_model, lstm_model, sequence_length=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # RF predictions
    rf_pred = rf_model.predict(X)
    
    # Prepare data for LSTM
    dataset = TimeSeriesDataset(X, pd.Series(np.zeros(len(X))), sequence_length)
    loader = DataLoader(dataset, batch_size=32)
    
    # LSTM predictions
    lstm_model.eval()
    lstm_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            lstm_preds.extend(outputs.cpu().numpy())
    
    # Average the predictions
    lstm_preds = np.array(lstm_preds).flatten()
    ensemble_predictions = (rf_pred[sequence_length:] + lstm_preds) / 2
    return ensemble_predictions, rf_pred[sequence_length:], lstm_preds

if __name__ == '__main__':
    # Load and prepare data
    data = pd.read_csv(final_data_path)
    feature_cols = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3',
                   'temperature_2m', 'relative_humidity_2m', 'rain',
                   'wind_speed_10m', 'wind_direction_10m',
                   'soil_temperature_0_to_7cm', 'soil_moisture_0_to_7cm']

    X = data[feature_cols]
    y = data['aqi']

    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    with open(rf_model_path, 'wb') as f:
        pickle.dump(rf_model, f)

    # Prepare LSTM datasets
    sequence_length = 24
    batch_size = 32

    train_dataset = TimeSeriesDataset(X_train_scaled, y_train, sequence_length)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test, sequence_length)

    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train LSTM
    print("\nTraining LSTM model...")
    lstm_model = LSTMModel(input_size=len(feature_cols))
    train_losses, val_losses = train_lstm(train_loader, val_loader, lstm_model)

    # Plot LSTM losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('LSTM Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(lstm_loss_path)
    plt.close()

    # Make predictions
    print("\nMaking ensemble predictions...")
    train_preds, train_rf_preds, train_lstm_preds = make_ensemble_predictions(
        X_train_scaled, rf_model, lstm_model)
    test_preds, test_rf_preds, test_lstm_preds = make_ensemble_predictions(
        X_test_scaled, rf_model, lstm_model)

    # Calculate metrics for individual models and ensemble
    def calculate_metrics(y_true, y_pred):
        return {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred)
        }

    # Training metrics
    train_metrics = {
        'ensemble': calculate_metrics(y_train[24:], train_preds),
    }

    # Test metrics
    test_metrics = {
        'ensemble': calculate_metrics(y_test[24:], test_preds),
    }

    # Print metrics
    print("\nModel Performance Metrics:")
    for model_name in ['ensemble']:
        print(f"\n{model_name.upper()} Model:")
        print("\nTraining Metrics:")
        print(f"R² Score: {train_metrics[model_name]['r2_score']:.4f}")
        print(f"RMSE: {train_metrics[model_name]['rmse']:.4f}")
        print(f"MAE: {train_metrics[model_name]['mae']:.4f}")
        
        print("\nTest Metrics:")
        print(f"R² Score: {test_metrics[model_name]['r2_score']:.4f}")
        print(f"RMSE: {test_metrics[model_name]['rmse']:.4f}")
        print(f"MAE: {test_metrics[model_name]['mae']:.4f}")

    # Plot feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance['feature'], feature_importance['importance'])
    plt.xticks(rotation=45, ha='right')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(feature_importance_path)
    plt.close()

    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'timestamp': data.loc[y_test[24:].index, 'timestamp'].values,
        'state': data.loc[y_test[24:].index, 'state'].values,
        'actual_aqi': y_test[24:].values,
        'ensemble_predicted_aqi': test_preds,
        'ensemble_absolute_error': np.abs(y_test[24:].values - test_preds)
    })
    predictions_df.to_csv(test_predictions_path, index=False)

    # Save metrics to JSON
    all_metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': feature_importance.to_dict(orient='records'),
        'training_params': {
            'sequence_length': sequence_length,
            'batch_size': batch_size,
            'feature_columns': feature_cols,
            'rf_params': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5
            }
        }
    }

    with open(model_metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print("\nResults have been saved to:")
    print("- test_predictions.csv (predictions)")
    print("- model_metrics.json (performance metrics)")
    print("- lstm_loss.png (loss curves)")
    print("- feature_importance.png (Random Forest feature importance)")