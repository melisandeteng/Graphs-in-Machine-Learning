function [accuracy] = offline_face_recognition()
% [accuracy] = offline_face_recognition()
%     a skeleton function to test offline face recognition, needs to be completed

cc = cv.CascadeClassifier('haarcascade_frontalface_default.xml');

EXTR_FRAME_SIZE = 96;

X = zeros(100,EXTR_FRAME_SIZE^2);
Y = zeros(100,1);

for i = 0:9
    for j = 1:10
        im = imread(sprintf('data/10faces/%d/%02d.jpg',i,j));
        box = cc.detect(im);
        top_face.area = 0;
        frame.width = size(im,2);
        frame.height = size(im,1);
        for z = 1:length(box)
            rect.x = box{z}(1);
            rect.y = box{z}(2);
            rect.width = box{z}(3);
            rect.height = box{z}(4);
            face_area = rect.width*rect.height;
            if (face_area > top_face.area)
                top_face.area = face_area;
                top_face.x = box{z}(1);
                top_face.y = box{z}(2);
                top_face.width = box{z}(3);
                top_face.height = box{z}(4);
            end
        end

        gray_im = cv.cvtColor(im, 'BGR2GRAY');
        gray_face = gray_im( top_face.y:top_face.y + top_face.height, top_face.x:top_face.x + top_face.width);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Apply preprocessing to balance the image (color/lightning), such    %
        % as filtering (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and %
        % equalization (cv.equalizeHist).                                     %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        gray_face = cv.GaussianBlur(gray_face);
        %gray_face = cv.equalizeHist(gray_face);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %resize the face and reshape it to a row vector
        gray_face = cv.resize(gray_face, [EXTR_FRAME_SIZE, EXTR_FRAME_SIZE]);
        if i<2
          subplot(2,10,i*10 + j);
          title(strcat(num2str(i), num2str(j)));
          imshow(gray_face);
        end
        X(sub2ind([10,10],i+1,j),:) = gray_face(:)';
        Y(sub2ind([10,10],i+1,j),1) = i+1;
    end
end


% if you want to plot the dataset, set the following variable to 1

plot_the_dataset = 0;

if plot_the_dataset == 1

    figure()
    for i=1:20
        subplot(2,10,i);
        h = imshow(reshape(X(i,:),EXTR_FRAME_SIZE,EXTR_FRAME_SIZE));
        title(num2str(i))
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% select 4 random labels PER PERSON and reveal them             %
% Y_masked: (n x 1) masked label vector, where entries Y_i      %
%       takes a value in [1..num_classes] if the node is        %
%       labeled, or 0 if the node is unlabeled (masked)         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y_masked = zeros(100,1);
for i = 0 :9
  Y_masked(i*10 + 1:(i+1) * 10) =mask_labels(Y(i*10 + 1:(i+1) * 10), 4);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% choose the experiment parameter                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

graph_param.graph_type = 'knn' ; %'knn' or 'eps'
graph_param.graph_thresh = 9; % the number of neighbours for the graph or the epsilon threshold
graph_param.sigma2 = mean(var(X)); % exponential_euclidean's sigma^2

laplacian_param.normalization = 'unn'; %either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization
laplacian_param.regularization = 0.1; %regularization to add to the laplacian (\gamma_g)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%compute hfs solution using either soft_hfs.m or hard_hfs.m    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%labels = hard_hfs(X, Y_masked, graph_param, laplacian_param);
labels = soft_hfs(X, Y_masked, 0.95, 0.05, graph_param, laplacian_param);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

set(figure(), 'units', 'centimeters', 'pos', [0 0 20 10]);
%
subplot(1,2,1);
imagesc(reshape(Y,10,10));
title('True labels')
%
subplot(1,2,2);
imagesc(reshape(labels,10,10));
title('HFS classification')

accuracy = mean(Y == labels);

end
