 
input_model_file = "PMnet.mat";

[~, basename, ext] = fileparts(input_model_file);
output_model_file = [basename ".onnx"];

net = load(input_model_file);
varnames = fieldnames(net);
net = net.(varnames{1});

% Needs to download this function from:
% https://mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format
exportONNXNetwork(net, output_model_file, 'BatchSize', 1);