function [Y] = spectral_clustering(L, chosen_eig_indices, num_classes)
%  [Y] = spectral_clustering(L, chosen_eig_indices, num_classes)
%      a skeleton function to perform spectral clustering, needs to be completed
%
%  Input
%  L:
%      Graph Laplacian (standard or normalized)
%  chosen_eig_indices:
%      indices of eigenvectors to use for clustering
%  num_classes:
%      number of clusters to compute (defaults to 2)
%
%  Output
%  Y:
%      Cluster assignments

if nargin < 3
    num_classes = 2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute eigenvectors                                          %
% U = (n x n) eigenvector matrix                                %
% E = (n x n) eigenvalue diagonal matrix                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[U,E] = eig(L);

%eig vs.eigs no difference observed in time taken...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[eigenvalues_sorted,reorder] = sort(diag(E));
disp(eigenvalues_sorted);

U = U(:,reorder(chosen_eig_indices));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the clustering assignment from the eigenvector        %
% Y = (n x 1) cluster assignments [1,2,...,c]                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%this corresponds to Normalized spectral clustering according to Shi and Malik (2000) L_rw
%Y = kmeans(U,num_classes);

%this corresponds to Normalized spectral clustering according to Ng L_sym
%U = U ./ norm(U);
%Y = kmeans(U,num_classes);

Y = kmeans(U,num_classes);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
