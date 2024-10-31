%%%%%%%%read dataset
dataset = xlsread('Dataset.xlsx');
%%%%%%%%%%%%%%%%%%%%
data_size = 0.8;
outdimension = 1;
nums = size(dataset, 1);
dataset = dataset(randperm(nums), :);
numt = round(data_size * nums);
f = size(dataset, 2) - outdimension;
%%%%%%%%%%%%%%%%%%%%
Itrain = dataset(1: numt, 1: f)';
Otrain = dataset(1: numt, f + 1: end)';
M = size(Itrain, 2);
%%%%%%%%%%%%%%%%%%%%
Itest = dataset(numt + 1: end, 1: f)';
Otest = dataset(numt + 1: end, f + 1: end)';
N = size(Itest, 2);
%%%%%%%%%%%%%%%%%%%%
[p_train, ps_input] = mapminmax(Itrain, 0, 1);
p_test = mapminmax('apply', Itest, ps_input);
%%%%%%%%%%%%%%%%%%%%
[ttrain, ps_output] = mapminmax(Otrain, 0, 1);
t_t = mapminmax('apply', Otest, ps_output);
%%%%%%%%%%%%%%%%%%%%
p_train = p_train';
p_test = p_test';
ttrain = ttrain'; 
t_t = t_t';
%%%%%%%%%%%%%%%%%%%%
trees = 6;
leaf  = 0.01;
OOBPrediction = 'on';
OOBPredictorImportance = 'on';
Method = 'regression';
net = TreeBagger(trees, p_train, ttrain, 'OOBPredictorImportance', OOBPredictorImportance,...
    'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;
%%%%%%%%%%%%%%%%%%%%
t_simulate1 = predict(net, p_train);
t_simulate2 = predict(net, p_test );
%%%%%%%%%%%%%%%%%%%%
T_simulate1 = mapminmax('reverse', t_simulate1, ps_output);
T_simutate2 = mapminmax('reverse', t_simulate2, ps_output);
%%%%%%%%%%%%%%%%%%%%
error1 = sqrt(sum((T_simulate1' - Otrain).^2) ./ M);
error2 = sqrt(sum((T_simutate2' - Otest ).^2) ./ N);


