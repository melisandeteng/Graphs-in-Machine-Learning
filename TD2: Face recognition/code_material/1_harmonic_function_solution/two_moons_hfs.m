function [accuracy] = two_moons_hfs()
% [accuracy] = two_moons_hfs()
% a skeleton function to perform HFS, needs to be completed


% load the data

try 
cd '/home/student/Desktop/TP2'
load_td_path;
end
%in_data = load('data_2moons_hfs.mat');
%in_data = load('data_2moons_hfs_large.mat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% at home, try to use the larger dataset (question 1.2)         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

in_data = load('data_2moons_hfs_large.mat');
%fprintf('using larger dataset!\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = in_data.X;
Y = in_data.Y;

% automatically infer number of labels from samples
num_classes = length(unique(Y));
num_samples = length(Y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose the experiment parameter                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_type = 'eps'; %'knn' or 'eps'
graph_param.graph_thresh = 0.888; % the number of neighbours for the graph or the epsilon threshold
%5 gives errors for hfs_large with mean(var(X))
graph_param.sigma2 =mean(var(X)); % exponential_euclidean's sigma^2
disp(graph_param.sigma2);
laplacian_param.normalization = 'sym'; %either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
laplacian_param.regularization = 0.1%regularization to add to the laplacian (\gamma_g)

l = 4;% number of labeled (unmasked) nodes provided to the hfs algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mask labels
Y_masked =  mask_labels(Y, l);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute hfs solution using either soft_hfs.m or hard_hfs.m    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
labels =  hard_hfs(X, Y_masked, graph_param, laplacian_param);
%labels =  soft_hfs(X, Y_masked, 0.95,0.05, graph_param, laplacian_param);
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_classification(X, Y, graph_param, labels);
accuracy = mean(labels == Y);

disp(accuracy);
