function [new_label] = online_ssl_compute_solution(t, online_cover_state, Y, gamma_g)
% [new_label] = online_ssl_compute_solution(t, online_cover_state, Y, gamma_g)
%     Computes the HFS solution given an online cover and labels
%
% Input
% t:
%     the current time step
% online_cover_state:
%     the current cover state, returned by online_ssl_update_centroids.m
% Y:
%     (n x 1) label vector
%
%  gamma_g:
%      regularization constant
%
% Output
% new_label:
%     computed label for the sample received at time t

centroids = online_cover_state.centroids;
nodes_to_centroids_map = online_cover_state.nodes_to_centroids_map;
centroids_to_nodes_map = online_cover_state.centroids_to_nodes_map;

num_centroids = size(centroids,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose the experiment parameter                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_type = 'knn'; %'knn' or 'eps'
graph_param.graph_thresh =10; % the number of neighbours for the graph or the epsilon threshold
graph_param.sigma2 = 1; % exponential_euclidean's sigma^2

laplacian_regularization = 0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% use the build_similarity_graph function to build the          %
% graph W, the similarity matrix on the centroids               %
% W_tilda_q: (num_centroids x num_centroids) dimensional matrix %
% representing the adjacency matrix of the graph                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W_tilda_q = build_similarity_graph(centroids, graph_param);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build the multiplicites vector v and the normalized W_q       %
% v: (num_centroids x 1) vector                                 %
% V: (num_centroids x num_centroids) diagonal matrix built      %
%       from v                                                  %
% W_q: (num_centroids x num_centroids) dimensional matrix       %
% representing the normalized adjacency matrix of the graph     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:num_centroids
    v(i) = sum(nodes_to_centroids_map == centroids_to_nodes_map(i));
end

V = diag(v);
W_q = V*W_tilda_q*V;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the quantized laplacian                               %
% L: (num_centroids x num_centroids) quantized laplacian        %
% Q: (num_centroids x num_centroids) dimensional matrix         %
%     representing the laplacian with gamma_g regularization    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Q and not L because it does everything in build_laplacian_regularized
degres = sum(W_q,1);

D = diag(transpose(degres));

n = length(degres);
%unn laplacian was chosen

L = D - W_q;
%L = eye(n) - D^(-1)*W_q;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the target y for the linear system eigenvector        %
% Y_mapped = (num_centroids x 1) labels of the nodes that are   %
%               currently a representative for a centroid       %
% y = (l x 1) target vector with {+1,-1} entries                %
% l_idx = (l x 1) vector with indices of labeled nodes          %
% u_idx = (u x 1) vector with indices of unlabeled nodes        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Y_mapped = Y(centroids_to_nodes_map);

l_idx = find(Y_mapped~=0);
u_idx = find(Y_mapped==0);

l = length(l_idx);

y = Y_mapped(l_idx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = y(:); %make sure it's a column vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the hfs solution
% f_l = (l x 1) hfs solution for labeled nodes                  %
% f_u = (u x 1) hfs solution for unlabeled nodes                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f_l = y;
Q_uu = L(u_idx, u_idx) + laplacian_regularization*V(u_idx,u_idx);
W_ul = W_q(u_idx, l_idx);

f_u= Q_uu \ (W_ul * f_l);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the hfs solution           %
% new_label:  {+1, -1, 0} class assignment for new sample       %
% new_label_confidence: confidence for new sample assignment    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

new_label_vector = f_u(centroids_to_nodes_map(u_idx)==t);

new_label_confidence = abs(new_label_vector);
new_label = sign(new_label_vector);
disp('new_label');
disp(new_label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('label %d confidence %.5f time %d\n', new_label, new_label_confidence, t);
fflush(stdout);
