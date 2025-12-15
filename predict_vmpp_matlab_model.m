function vmpp = predict_vmpp_matlab_model(T, P_H2, P_O2)
%PREDICT_VMPP_MATLAB_MODEL Predicts Vmpp using neural network
%   vmpp = predict_vmpp_matlab_model(T, P_H2, P_O2)
%   
%   Inputs:
%       T     - Temperature (K)
%       P_H2  - Hydrogen pressure (bar)
%       P_O2  - Oxygen pressure (bar)
%   
%   Output:
%       vmpp  - Predicted maximum power point voltage
%
%   Network architecture: 3 -> 32 -> 16 -> 1 (ReLU activation)
%
%   Example: 
%       vmpp = predict_vmpp_matlab_model(328, 1.5, 1.0);

%#codegen
    persistent W1 b1 W2 b2 W3 b3 means stds
    
    if isempty(W1)
        data = coder.load('S:\Doctorat_Setif\nn_weights_matlab_model.mat');
        W1 = data.W1;
        b1 = data. b1;
        W2 = data.W2;
        b2 = data.b2;
        W3 = data.W3;
        b3 = data.b3;
        means = data.scaler_mean(:);
        stds = data.scaler_scale(:);
    end
    
    % Input features: [T, P_H2, P_O2]
    x = [T; P_H2; P_O2];
    
    % Normalize inputs
    x = (x - means) ./ stds;
    
    % Layer 1: Input -> Hidden1 (3 -> 32)
    z1 = W1' * x + b1;
    a1 = max(0, z1);  % ReLU activation
    
    % Layer 2: Hidden1 -> Hidden2 (32 -> 16)
    z2 = W2' * a1 + b2;
    a2 = max(0, z2);  % ReLU activation
    
    % Layer 3: Hidden2 -> Output (16 -> 1)
    z3 = W3' * a2 + b3;
    vmpp = z3(1);  % Output: Vmpp prediction
end