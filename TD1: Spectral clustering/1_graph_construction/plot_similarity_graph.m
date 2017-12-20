function [] = plot_similarity_graph()
%  [] = plot_similarity_graph()
%      a skeleton function to analyze the construction of the graph similarity
%      matrix, needs to be completed

% the number of samples to generate
num_samples = 300;

% the sample distribution function with the options necessary for
% the distribution
sample_dist = @two_moons;
dist_options = [0.5, 0.01]; % two_moons [radius of the moons,
%       variance of the moons]
% blobs: number of blobs, variance of gaussian
%                                    blob, surplus of samples in first blob

[X, Y] = get_samples(sample_dist, num_samples, dist_options);
%[X, Y] = two_moons(num_samples, dist_options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  choose the type of the graph to build and the respective     %
%  threshold and similarity function options                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_type = 'eps'; %'knn' or 'eps'
graph_param.graph_thresh = 0.75 % the number of neighbours for the graph or the epsilon threshold
graph_param.sigma2 = mean(var(X)); % exponential_euclidean's sigma^2


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the build_similarity_graph function to build the graph W  %
% W: (n x n) dimensional matrix representing                    %
%    the adjacency matrix of the graph                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W = build_similarity_graph(X, graph_param);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_graph_matrix(X,W);
title(['Two moons :' graph_param.graph_type ' graphthresh = ' num2str(graph_param.graph_thresh)  ]) ;
