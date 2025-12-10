function vmpp = predict_vmpp(T, Pfuel, Pair, Vfuel, Vair)
%#codegen
    persistent W1 b1 W2 b2 W3 b3 means stds
    
    if isempty(W1)
        data = coder.load('S:\Doctorat_Setif\nn_weights.mat');
        W1 = data.W1;
        b1 = data. b1;
        W2 = data.W2;
        b2 = data.b2;
        W3 = data.W3;
        b3 = data.b3;
        means = data.scaler_mean(:);
        stds = data.scaler_scale(:);
    end
    
    x = [T; Pfuel; Pair; Vfuel; Vair];
    x = (x - means) ./ stds;
    
    z1 = W1' * x + b1;
    a1 = max(0, z1);
    
    z2 = W2' * a1 + b2;
    a2 = max(0, z2);
    
    z3 = W3' * a2 + b3;
    vmpp = z3(1);
end
