function [labels, accuracy] = iterative_hfs(T)
% function [labels, accuracy] = iterative_hfs(t)
% a skeleton function to perform HFS, needs to be completed
%  Input
%  T:
%      number of iterations to use for the iterative propagation

%  Output
%  labels:
%      class assignments for each (n) nodes
%  accuracy

% load the data

in_data = load('data/data_iterative_hfs_graph.mat');

W = in_data.W;
Y = in_data.Y;
Y_masked =  in_data.Y_masked;

num_samples = size(W,1);
num_classes = sum(unique(Y) ~= 0);
iteration = 0;
%regulatization parameter

gamma = 0.01;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the target y for the linear system                       %
% y = (l x num_classes) target vector                              %
% l_idx = (l x num_classes) vector with indices of labeled nodes   %
% u_idx = (u x num_classes) vector with indices of unlabeled nodes %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l_idx = find(Y_masked~=0);
u_idx = find(Y_masked ==0);

y = Y(l_idx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the hfs solution, using iterated averaging            %
% remember that column-wise slicing is cheap, row-wise          %
% expensive and that W is already undirected                    %
% f_l = (l x num_classes) hfs solution for labeled              %
% f_u = (u x num_classes) hfs solution for unlabeled            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f_l = y;

f = zeros(num_samples,1);
f(l_idx)= f_l;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Equation 1 in the report, substructing terms corresponding to j = i, 
%very costly in time -NOT TO BE USED IN PRACTICE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic
%for k = 1:T
%   fl = (f~=0);
%  for i = 1 : length(u_idx)
%    
%    sum_w =  (W(:,u_idx(i))'* fl);%- W(u_idx(i), u_idx(i))*fl(u_idx(i));
%    sum_f = (f.*fl)'*W(:,u_idx(i));%- f(u_idx(i))*W(u_idx(i), u_idx(i)) ;
%    if sum_w ~=0
%      f(u_idx(i)) = sum_f /sum_w;
%    else
%      f(u_idx(i)) =0;
%    end
%  end
%end
%toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Equation 2 in the report, much faster iterative label propagation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
for k = 1:T
  fl = (f~=0);
  
  sink = (rand(size(u_idx))>gamma| fl(u_idx));
  
  weights = W(:,u_idx)'* fl;
  sum_w = zeros(size(u_idx));
  sum_w(weights ~=0) = 1./weights(weights~=0);
  sum_w(weights ==0) = 0;
  f(u_idx) =  (sink) .*(W(:,u_idx)'* f ).*sum_w;
  labels = zeros(num_samples,1);
  
  f_u = f(u_idx);

toc
end


f_u = f(u_idx);
f_u(f_u <= 1.5) = 1;
f_u(f_u > 1.5) = 2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the labels assignment from the hfs solution           %
% label: (n x 1) class assignments [1,2,...,num_classes]        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels = zeros(num_samples,1);
labels(l_idx)=f_l;
labels(u_idx)=f_u;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

accuracy = mean(labels == Y);
disp(accuracy);

