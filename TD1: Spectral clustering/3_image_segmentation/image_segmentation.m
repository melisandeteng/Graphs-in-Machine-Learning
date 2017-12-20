function [] = image_segmentation(input_img, input_ext)
%%  [] = image_segmentation(input_img, input_ext)
%%      a skeleton function to perform image segmentation, needs to be completed
%%  Input
%%  input_img:
%%      (string) name of the image file, without extension (e.g. 'four_elements')
%%  input_ext:
%%      (string) extension of the image file (e.g. 'bmp')
%   numclusters 
%
X = double(imread(input_img,input_ext));
X = reshape(X,[],3);
disp(size(X));
im_side = sqrt(size(X,1));
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Y_rec should contain an index from 1 to c where c is the      %
%% number of segments you want to split the image into           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
graph_param.graph_type = 'eps';
%we choose to build a densely connected graph. 
graph_param.graph_thresh = 0.05; % the number of neighbours for the graph
graph_param.sigma2 = 100;
L = build_laplacian(X, graph_param, 'rw') ;

%the parameters are entered for the fruit_salad image  we want to have 4 clusters. so we will choose numclusters  = 4
Y_rec = spectral_clustering(L,1:4, 4);

%for the four_elements we can choose 6 clusters and the second parameter
% for spectral_clustering 1:6 (this will count the shadow as a cluster)
% but we were lucky once to get the background and the shadows counted as one cluster 
%with spectral_clustering(L,1:6, 5) and good clustering ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
figure()
%
subplot(1,2,1);
imagesc(imread(input_img,input_ext));
axis square;
%
subplot(1,2,2);
imagesc(reshape(Y_rec,im_side,im_side));
