
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras_tuner import RandomSearch
import numpy as np
import matplotlib.pyplot as plt
from analysis import calculate_silhouette_score

def train_autoencoder(features, input_dim, epochs=100, batch_size=32, num_trials=10, save_history=True, history_path='autoencoder_history.json'):
    from autoencoder_model import AutoencoderHyperModel
    hypermodel = AutoencoderHyperModel(input_dim)

    # Split data into training and validation sets
    X_train, X_val = train_test_split(features, test_size=0.2, random_state=42)

    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=num_trials,
        seed=42,
        executions_per_trial=2,
        directory='autoencoder_tuning',
        project_name='autoencoder')

    tuner.search(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val))
    best_model = tuner.get_best_models(num_models=1)[0]

    # Callbacks including TensorBoard
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ModelCheckpoint(filepath='best_autoencoder_model.h5', monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir='./logs')
    ]
    history = best_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, X_val), callbacks=callbacks, verbose=1)

    return best_model, history

def evaluate_model(model, features, labels=None):
    if np.isnan(features).any():
        raise ValueError("Features contain NaN values.")

    predictions = model.predict(features)
    reconstruction_error = mean_squared_error(features, predictions)

    if labels is not None:
        silhouette_score_value = calculate_silhouette_score(features, labels)
        print(f"Silhouette Score: {silhouette_score_value}")
    else:
        silhouette_score_value = None
        print("No labels provided for silhouette score calculation.")

    print(f"Reconstruction Error (MSE): {reconstruction_error}")

    return reconstruction_error, silhouette_score_value
