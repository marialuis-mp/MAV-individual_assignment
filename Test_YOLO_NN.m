%--------------------------------------------------------------------------
% Test_YOLO_NN: 
% Tests the neural network, shows detection on training data and test data,
% evaluates the detection frequency, calculates and plots the ROC curves
%--------------------------------------------------------------------------
clear;
clc;
close all;
load('NN_YOLO.mat');
%% Parameters / declaring variables
% - - - - - Parameters - - - - - 
show_pictures = 0; % set to 1 to show pictures, 0 to not show
show_training_pictures = 0; % set to 1 to show training pictures
Threshold_IoU_Number_of_points = 6; %number of ROC curves
Number_of_points_ROC = 51; %number of points in each ROC curve
Score_threshold = 0.5333; %Score above a detection is shown in images
% - - - Declaring Variables - - - 
Time = zeros(2,1);
FP = zeros(Threshold_IoU_Number_of_points,Number_of_points_ROC);
TPR = zeros(Threshold_IoU_Number_of_points,Number_of_points_ROC);
%% Show Results - Training Data
Time(1)=0;
for i = 1:height(training_data)
    %---------- Predict Output with NN -----------------------
    img = imread(training_data.Var1{i});
    tic;
    [bbox, score,label] = detect(detector, img);
    Time(1)=Time(1)+toc;
    if ~isempty(score) && show_training_pictures == 1% When a gate is detected: 
        %---------- Choose Threshold of confidence----------------
        idx = find(score>Score_threshold);
        if isempty(idx)
            [score, idx] = max(score);  
        end
        score=score(idx);
        bbox = bbox(idx, :);
        %---------- Label ----------------------------------------
        label_str = cell(length(score),1);
        for ii=1:length(score)
            label_str{ii} = ['Confidence =', num2str(score(ii))];
        end
        %---------- Show Results ---------------------------------        
        figure
        detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, label_str);
        imshow(detectedImg)
    end
end
%% Show Results - Test Data
Time(2)=0;
Time_max=zeros(1,height(test_data));
for i = 1:height(test_data)
    %---------- Predict Output with NN -----------------------
    img = imread(test_data.Var1{i});
    tic;
    [bbox, score,label] = detect(detector, img);
    Time(2)=Time(2)+toc;
    Time_max(i)=toc;
    if ~isempty(score) && show_pictures == 1% When a gate is detected: 
        %---------- Choose Threshold of confidence----------------
        idx = find(score>Score_threshold);
        if isempty(idx)
            [score, idx] = max(score);  
        end
        score=score(idx);
        bbox = bbox(idx, :);
        %---------- Label ----------------------------------------
        label_str = cell(length(score),1);
        for ii=1:length(score)
            label_str{ii} = ['Confidence =', num2str(score(ii))];
        end
        %---------- Show Results ---------------------------------        
        figure
        detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, label_str);
        imshow(detectedImg)
    end
end
%% Frequency Analysis
% Average Times:
Time(1)=Time(1) ./ height(training_data);
Time(2)=Time(2) ./ height(test_data);
% Maximum time for test data
maximum_time = max(Time_max);
% Frequencies:
frequency = 1 ./ Time;
fprintf('\nFrequency training: %f Hz \nFrequency test: %f Hz', frequency(1), frequency(2));
frequency= 1 / maximum_time;
fprintf('\nMinimum test frequency: %f Hz\n', frequency);

%% Intersection Over Union
% 1. Get the table with test data and its points
organized_file = organize_file(File, number_rows);

test_data_points = table;
for j = 1:length(random_sample)-1
    %test_data_points.Var1{j} = organized_file.Var1{random_sample(j)};
    test_data_points{j,:} = organized_file{random_sample(j),2:end};
end

% 2. For each Threshold of IoU:
for j = 1:Threshold_IoU_Number_of_points
    TP = zeros(1,Number_of_points_ROC);
    FN = zeros(1,Number_of_points_ROC);
    for i = 1:height(test_data_points)
        % 2.a Read each image
        img = imread(test_data.Var1{i});
        [bbox, score,label] = detect(detector, img);
        for k = 1:Number_of_points_ROC
            %------------ Choose Threshold of confidence ------------------
            Confidence_threshold = (k-1) / (Number_of_points_ROC-1);
            idx_k = find(score>Confidence_threshold);
            score_k = score(idx_k);
            bbox_k = bbox(idx_k, :);

            %-------------- Calculate the IoU Matrix ----------------------
            J=size(test_data_points.Var1{i},1); % number of gates
            K=size(bbox_k,1); %number of bbox predicted
            IoU = calculate_IoU(i,J,K,bbox_k,test_data_points);
            
            %------ Calculate the Number of TP, FP, FN in this image ------
            if(K==0) %no predictions at all
                FN(k) = FN(k) + J;  % All gates are FN 
            else
                Threshold_IoU = (j-1) / (Threshold_IoU_Number_of_points-1)*0.5;
                [TP_i,FP_i,FN_i]= Calculate_TP_FP_FN(IoU, J, K, Threshold_IoU);

                TP(k) = TP(k) + TP_i;
                FP(j,k) = FP(j,k) + FP_i;
                FN(k) = FN(k) + FN_i;
            end
        end

        if ~isempty(score) && show_pictures == 1% When a gate is detected: 
            %---------- Label ----------------------------------------
            label_str = cell(length(score),1);
            for ii=1:length(score)
                label_str{ii} = ['Confidence =', num2str(score(ii))];
            end
            %---------- Show Results ---------------------------------        
            figure
            detectedImg = insertObjectAnnotation(img, 'rectangle', bbox, label_str);
            imshow(detectedImg)
        end
    end
    TPR(j,:) = TP./(TP+FN);
end

%% Show graph
figure()
legend_graph = zeros(1,Threshold_IoU_Number_of_points); %create legend
% Plot each Curve
for j = 3:Threshold_IoU_Number_of_points
    legend_graph(1, j) = (j-1) / (Threshold_IoU_Number_of_points-1)*0.5;
    plot(FP(j,:)/FP(j,1),TPR(j,:));
    %plot(FP(j,:),TPR(j,:));
    hold on;
end
% Titles and labels
title('ROC Curves for various IoU thresholds');
xlabel('False Positives (normalized)');
ylabel('True Positive Rate');
legend_graph = string(legend_graph(3:end));
legend(legend_graph,'Location','southeast');
hold off