function [eig_ind] = choose_eig_function(eigenvalues)
%  [eig_ind] = choose_eig_function(eigenvalues)
%     chooses indices of eigenvalues to use in clustering
%
% Input
% eigenvalues:
%     eigenvalues sorted in ascending order
%
% Output
% eig_ind:
%     the indices of the eigenvectors chosen for the clustering
%     e.g. [1,2,3,5] selects 1st, 2nd, 3rd, and 5th smallest eigenvalues

%eigengap heuristic
eigen_jumps = (eigenvalues(2:end) - eigenvalues(1:(end-1))) ./ eigenvalues(2:end) ;

[~,before_jump] = max(eigen_jumps(min(find(eigenvalues>(10^(-13))):end))) ;
eig_ind = 1:(min(find(eigenvalues>10^(-13)))+before_jump-1);
%we start looking at the gap between eigenvalues after those that are null. 