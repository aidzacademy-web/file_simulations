function result = call_predict_vmpp(T, Pfuel, Pair, Vfuel, Vair)
    % Use persistent variables to avoid re-initialization
    persistent python_initialized;
    persistent pytorch_module;
    
    if isempty(python_initialized)
        % One-time initialization
        py.importlib.invalidate_caches();
        
        % Add path only once
        python_path = 'S:\Doctorat_Setif\Thése\Fuel-cell\5th_presentation\second_try';
        if count(py.sys.path, python_path) == 0
            insert(py.sys.path, int32(0), python_path);
        end
        
        % Import module once
        pytorch_module = py.importlib.import_module('pytorch_model');
        
        python_initialized = true;
        fprintf('Python initialized (one-time setup)\n');
    end
    
    % Call the Python function
    result = pytorch_module.predict_vmpp(T, Pfuel, Pair, Vfuel, Vair);
end