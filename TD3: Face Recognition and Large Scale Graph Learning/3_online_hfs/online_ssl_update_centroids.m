function [online_cover_state] = online_ssl_update_centroids(t, face, online_cover_state, num_labels)
% [online_cover_state] = online_ssl_update_centroids(t, face, online_cover_state, num_labels)
%     Updates the online cover when a new sample arrives
%
% Input
% t:
%     the current time step
% face:
%     the new sample
% online_cover_state:
%     the current cover state, returned by online_ssl_update_centroids.m
% num_labels:
%     the initial number of labeled samples
%
% Output
% online_cover_state:
%     the updated cover state

if nargin < 4
    num_labels = 20;
end
unknown = false;
% unpack state
centroids = online_cover_state.centroids;
max_num_centroids = online_cover_state.max_num_centroids;
nodes_to_centroids_map = online_cover_state.nodes_to_centroids_map;
centroids_to_nodes_map = online_cover_state.centroids_to_nodes_map;
taboo = online_cover_state.taboo;
R = online_cover_state.R;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check if unknown face
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%if the face is too different from the two for which pictures have been stored 
%we want to ignore it 
distance_threshold = 200;
graph_param.graph_type = 'eps';
graph_param.graph_thresh = 0.;
graph_param.sigma2 = 1;


size_face_added = size(centroids,1)+1;
similarities_face = build_similarity_graph([centroids; face], graph_param);
distances_face = 1./(similarities_face + 1e-30);
distances_face(logical(eye(size_face_added))) = 1e30;

if min(distances_face(size_face_added,:)) > distance_threshold
  unknown = true;
  printf("Unknown face at time %i  ", t);
  printf("To be ignored !");
  fflush(stdout);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if (size(centroids,1) == max_num_centroids)


    % build similarity matrix for all couples of centroids
    graph_param.graph_type = 'eps';
    graph_param.graph_thresh = 0.;
    graph_param.sigma2 = 1;

    [similarities] = build_similarity_graph(centroids, graph_param);

    %transform similarities into distances
    distances = 1./(similarities + 1e-30);

    % set taboo nodes and self loops as infinitely distant
    % (minimally similar),
    % so they do not get merged

    distances(:, taboo) = 1e30;
    distances(taboo,:) = 1e30;

    distances(logical(eye(max_num_centroids))) = 1e30;

    % find the position of the two closest vertices
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % find the edge with the minimum distance (maximum similarity)  %
    % and assign its row index to c_rep, and column index to c_add  %
    % in other words:                                               %
    % c_rep is the row i of the edge found                          %
    % c_add is the column j of the edge found                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [dist_min,c_rep] = min(min(distances, [],2));
    [~,c_add] = min(distances(c_rep,:));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % if c_rep is not taboo and is not part of the labeled nodes

    if ~taboo(c_rep) && c_rep > num_labels

        % if c_add represents more nodes then c_rep, swap c_add and c_rep
        % so that the larger centroid gets larger
        %
        % the nested if syntax is to avoid problems with comments, newlines and
        % auto-reformatting of __some__ editors

        if ( sum( nodes_to_centroids_map == centroids_to_nodes_map(c_add) ) 
             > sum( nodes_to_centroids_map == centroids_to_nodes_map(c_rep) ))
            auxv = c_rep;
            c_rep = c_add;
            c_add = auxv;
        end
    end

    if (R == 0)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % initialize R                                                  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        R = 1.00001*dist_min;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        taboo = [true(1, num_labels), false(1, max_num_centroids - num_labels)];
        fprintf('R = %.6f\n', R);
        fprintf('  %i quantization vertices\n', length(centroids_to_nodes_map));
    elseif (dist_min> R)
        % double R
        R = 1.5 * R;
        
        taboo = [true(1, num_labels), false(1, max_num_centroids - num_labels)];
        fprintf('R = %.6f\n', R);
        fprintf('  %i quantization vertices\n', length(centroids_to_nodes_map));
    end
  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update data structures                                        %
    % find all nodes currently assigned to the c_add centroid,      %
    % and update their nodes_to_centroids_mapping to assign them    %
    % to the c_rep centroid instead.                                %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if unknown
      nodes_to_centroids_map(t) = c_add;
    else
    %data structure is only updated if the face is known
      nodes_to_centroids_map(nodes_to_centroids_map == c_add) = c_rep;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % we don't want to instantly remerge (at the next iteration)
    % something we already merged
    taboo(c_rep) = true;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update data structures                                        %
    % the centroids_to_nodes_map of c_add is updated to point to    %
    % the newly added node, and the data in the centroids matrix    %
    % needs to be updated accordingly.                              %
    % lastly, remember to assign the newly added node to the        %
    % newly added centroid in the correct mapping structure
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    centroids_to_nodes_map(c_add)= t;
    centroids(c_add,:) = face;
   nodes_to_centroids_map(t) =t;
  end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

else
    
    taboo(t) = false;
    
    if ~unknown
    % until we hit max_num_centroids we don't need to group nodes,
    % each node is added as his own centroid
    
      centroids_to_nodes_map = [centroids_to_nodes_map, t];
      nodes_to_centroids_map(t) = t;
      centroids = [centroids;face];
      centroids = centroids;
    end
end

% repack state
online_cover_state.centroids = centroids;
online_cover_state.max_num_centroids = max_num_centroids;
online_cover_state.nodes_to_centroids_map = nodes_to_centroids_map;
online_cover_state.centroids_to_nodes_map = centroids_to_nodes_map;
online_cover_state.taboo = taboo;
online_cover_state.R = R;
