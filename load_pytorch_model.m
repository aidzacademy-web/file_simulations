function Vmpp = load_pytorch_model(T, Pfuel, Pair, Vfuel, Vair)
    % Declare the external function as extrinsic
    coder.extrinsic('call_predict_vmpp');
    
    % Pre-allocate output (important for code generation)
    Vmpp = 0.0;
    
    % Call the external function
    result = call_predict_vmpp(T, Pfuel, Pair, Vfuel, Vair);
    
    % Convert to MATLAB double
    Vmpp = double(result);
end