import torch
import joblib
import numpy as np
import torch.nn as nn
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # Réseau de neurones à trois couches
        self.net = nn.Sequential(
            nn.Linear(5, 16),  # Input layer: 9 features -> 16 neurons
            nn.ReLU(),
            nn.Dropout(0.3),   # Dropout after ReLU

            nn.Linear(16, 8),  # Hidden layer: 16 -> 8 neurons
            nn.ReLU(),
            nn.Dropout(0.3),   # Dropout after ReLU

            nn.Linear(8, 1)    # Output layer: 8 -> 1 neuron
        )
    
    def forward(self, x):
        return self.net(x)

# Define paths (adjust as needed)
MODEL_PATH = r"S:\Doctorat_Setif\Thése\Fuel-cell\5th_presentation\second_try\model.pth"
SCALER_PATH = r"S:\Doctorat_Setif\Thése\Fuel-cell\5th_presentation\second_try\scaler.pkl"

# Instantiate the model and load the state dictionary
model = RegressionModel()
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()  # Set the model to evaluation mode

# Load the scaler for normalization
scaler = joblib.load(SCALER_PATH)

def predict_vmpp(T, Pfuel, Pair, Vfuel, Vair):
    """
    Predict Vmpp for given irradiance and temperature.
    
    Parameters:
        irradiance (float): The irradiance value.
        temperature (float): The temperature value.
        
    Returns:
        float: The predicted Vmpp value.
    """
    # Prepare input data (1x2 NumPy array)
    input_data = np.array([[T, Pfuel, Pair, Vfuel, Vair]], dtype=np.float32)
    
    # Normalize input using the scaler
    input_data = scaler.transform(input_data)
    
    # Convert input to PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    
    # Run inference
    with torch.no_grad():
        predicted_vmpp = model(input_tensor).numpy().flatten()[0]
    
    return float(predicted_vmpp)