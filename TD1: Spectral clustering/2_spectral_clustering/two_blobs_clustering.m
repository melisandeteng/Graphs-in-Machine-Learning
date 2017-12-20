function [] = two_blobs_clustering()
%  [] = two_blobs_clustering()
%       a skeleton function for questions 2.1,2.2

% load the data

in_data = load('data_2blobs.mat', '-mat');
X = in_data.X;
Y = in_data.Y;


% automatically infer number of labels from samples
num_classes = length(unique(Y));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose the experiment parameter                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_type = 'knn'; %'knn' or 'eps

sigma2 = 1;
similarities = exponential_euclidean(X, sigma2);
%max_tree = max_span_tree(similarities);
%%
%%minimum weight in max_tree is 0 so we exclude the value 0
%graph_thresh = min(setdiff(similarities.*max_tree,min(similarities.*max_tree)));
%%

graph_param.graph_thresh = 16; % the number of neighbours for the graph or the epsilon threshold
graph_param.sigma2 =sigma2; % exponential_euclidean's sigma^2

laplacian_normalization = 'unn'; %either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
chosen_eig_indices = 1:2; % indices of the ordered eigenvalues to pick

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% build laplacian
L =  build_laplacian(X, graph_param, laplacian_normalization);

Y_rec = spectral_clustering( L, chosen_eig_indices, num_classes);

plot_clustering_result(X, Y, L, Y_rec, kmeans(X, num_classes));
%title(num2str(chosen_eig_indices(1))'and' num2str(chosen_eig_indices(2)))