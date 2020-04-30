function [IoU] = calculate_IoU(i,J,K,bbox,test_data_points)
% For a image i, it receives the predicted bounding boxes (bbox), the num
% number of predictions K, the real gates position (test_data_points) and 
%the number of gates in that image J
%Returns a matrix IoU[JxK] with all intersection over union in which 
% IoU(jxk) corresponds to the IoU of gate j and bbox prediction k
IoU = zeros(J,K);

for j =1:J
    % Transforming prediction bbox in polygone
    gate = test_data_points.Var1{i}(j,:);
    X = [gate(1) gate(3) gate(5) gate(7)];
    Y = [gate(2) gate(4) gate(6) gate(8)];
    box_real = polyshape(X,Y);
    for k =1:K
        % Transforming prediction bbox in polygone
        X = [bbox(k,1) bbox(k,1) bbox(k,1)+bbox(k,3) bbox(k,1)+bbox(k,3)];
        Y = [bbox(k,2) bbox(k,2)+bbox(k,4) bbox(k,2)+bbox(k,4) bbox(k,2)];
        box_predicted = polyshape(X,Y);

        poly_union = union(box_predicted,box_real).Vertices;
        poly_intersection = intersect(box_predicted,box_real).Vertices;

        UNION = polyarea(poly_union(:,1),poly_union(:,2));
        INTERSECTION = polyarea(poly_intersection(:,1),poly_intersection(:,2));
        if INTERSECTION ~= 0
            IoU(j,k) = INTERSECTION/UNION;
        else
            IoU(j,k) = 0;
        end
    end
end

end

