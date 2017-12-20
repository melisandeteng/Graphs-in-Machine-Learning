function [] = how_to_choose_epsilon()
%  [] = how_to_choose_epsilon()
%       a skeleton function to analyze the influence of the graph structure
%       on the epsilon graph matrix, needs to be completed


% the number of samples to generate
num_samples = 200;

% the sample distribution function
sample_dist = @worst_case_blob;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the option necessary for worst_case_blob, try different       %
% values                                                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dist_options = 10 % read worst_case_blob.m to understand the meaning of
%                               the parameter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[X, Y] = get_samples(sample_dist, num_samples, dist_options);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the similarity function and the max_span_tree function    %
% to build the maximum spanning tree max_tree                   %
% sigma2: the exponential_euclidean's sigma2 parameter          %
% similarities: (n x n) matrix with similarities between        %
%              all possible couples of points                   %
% max_tree: (n x n) indicator matrix for the edges in           %
%           the maximum spanning tree                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% exponential_euclidean's sigma^2
sigma2 = mean(var(X));
similarities = exponential_euclidean(X, sigma2);
max_tree = max_span_tree(similarities);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set graph_thresh to the minimum weight in max_tree            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = similarities.*max_tree;
sizem= size(M);


graph_thresh = min(setdiff(M,0)*(1-(10^(-14))));

disp(graph_thresh);
%We want to exclude zero, and there is already strict inequality in build_similarity_graph 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_type = 'eps';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the build_similarity_graph function to build the graph W  %
% W: (n x n) dimensional matrix representing                    %
%    the adjacency matrix of the graph                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_thresh = graph_thresh; % the number of neighbours for the graph
graph_param.sigma2 = sigma2; % exponential_euclidean's sigma^2

W = build_similarity_graph(X, graph_param);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_graph_matrix(X,W);
title(['Worst case blob : graphthresh = ' num2str(graph_thresh) ' distoptions =' num2str(dist_options)]) ;