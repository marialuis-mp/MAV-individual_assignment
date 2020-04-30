%--------------------------------------------------------------------------
% Neural_network_YOLO: 
% Pre-processes data, builds and trains the neural network.
% At the end, saves a file with the trained network
%--------------------------------------------------------------------------
clear;
clc;
%% Parameters %%
% Tuneable parameters
Input_images_pixels = 90;
Percentage_for_test = 0.30;
num_anchor_boxes = 7;
training_epochs = 120;
% Constants and Variables declarations
inputSize = [Input_images_pixels Input_images_pixels 3];
number_Classes = 1;
Real_images_pixels = 360;
%% Extract data from excel %%
File = readtable('corners.csv');
number_rows = height(File);

%-------------------- Initialize variables for data ----------------------- 
data = table;
box = zeros(4,4);
j=1;
Number_of_pictures=1;
for i=1:number_rows-1
    % extract x1, y1, w and h from row
    x = min([File.Var4(i),File.Var6(i),File.Var2(i),File.Var8(i)]);
    y = min([File.Var7(i),File.Var9(i),File.Var3(i),File.Var5(i)]);
    w = max([File.Var4(i),File.Var6(i),File.Var2(i),File.Var8(i)])-x;
    h = max([File.Var7(i),File.Var9(i),File.Var3(i),File.Var5(i)])-y;
    box(j,1) = max(1,x-0.1*w);
    box(j,2) = max(1,y-0.1*h);
    w = 1.2 * w;
    h = 1.2 * h;
    box(j,3) = min(360 - box(j,1), w);
    box(j,4) = min(360 - box(j,2), h);
    if strcmp(File.Var1(i),File.Var1(i+1))
        j=j+1;
    else
        gate = zeros(j,4);
        for k = 1:j
            gate(k,:)=box(k,:);
        end
        j=1;
        data(Number_of_pictures, :)={File.Var1(i), {gate}};
        Number_of_pictures=Number_of_pictures+1;
    end
end
%--------------- Last image - needs to be outside the loop ----------------
i=i+1;
x = min([File.Var4(i),File.Var6(i),File.Var2(i),File.Var8(i)]);
y = min([File.Var7(i),File.Var9(i),File.Var3(i),File.Var5(i)]);
w = max([File.Var4(i),File.Var6(i),File.Var2(i),File.Var8(i)])-x;
h = max([File.Var7(i),File.Var9(i),File.Var3(i),File.Var5(i)])-y;
box(j,1) = max(1,x-0.1*w);
box(j,2) = max(1,y-0.1*h);
w = 1.2 * w;
h = 1.2 * h;
box(j,3) = min(360 - box(j,1), w);
box(j,4) = min(360 - box(j,2), h);
gate = zeros(j,4);
for k = 1:j
    gate(k,:)=box(k,:);
end
data(Number_of_pictures, :)={File.Var1(i), {gate}};

%% Split data between training and test %%
%--------------- Randomly selects the images for testing ------------------
n = round(Number_of_pictures * Percentage_for_test);
random_sample = randsample(Number_of_pictures,n);
random_sample = sort(random_sample);
%----- Divides the file with data between training and test ---------------
training_data = table;
test_data = table;
j=1;
for i = 1:Number_of_pictures
    if j < n
        if i == random_sample(j)
            test_data{j,:} = data{i,:};
            j = j + 1;
        else
            training_data{i-j+1,:} = data{i,:};
        end
    else
        training_data{i-j+1,:} = data{i,:};
    end
end

%% Neural Network Construction
%----------------------- Anchor Boxes Estimation --------------------------
training_data_estimate_AB = boxLabelDatastore(training_data(:,2));
[anchorBoxes, meanIoU] = estimateAnchorBoxes(training_data_estimate_AB, num_anchor_boxes);
%normalize the anchor boxes, due to image resize
anchorBoxes = round(anchorBoxes .* Input_images_pixels / Real_images_pixels);

%--------------------- Neural Network Architecture ------------------------
layers = [
    imageInputLayer(inputSize,'Name','input');
    
    convolution2dLayer([3 3], 16, 'Padding', 1,'Name','conv_1',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','norm_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer([3 3], 32, 'Padding', 1,'Name', 'conv_2',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','norm_2')
    reluLayer('Name','relu_2')
    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer([3 3], 64, 'Padding', 1,'Name','conv_3',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','norm_3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer(2, 'Stride',2,'Name','maxpool_3')
    
    convolution2dLayer([3 3], 128, 'Padding', 1,'Name','conv_4',...
    'WeightsInitializer','narrow-normal')
    batchNormalizationLayer('Name','BN4')
    reluLayer('Name','relu_4')
    ];

lgraph = layerGraph(layers);
lgraph = yolov2Layers(inputSize,number_Classes,anchorBoxes,lgraph,'relu_4');
%% Neural Network Training
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'Verbose',true,'MiniBatchSize',16,'MaxEpochs',training_epochs,...
    'Shuffle','every-epoch','VerboseFrequency',50, ...
    'DispatchInBackground',true,...
    'ExecutionEnvironment','auto');
    
[detector,info] = trainYOLOv2ObjectDetector(training_data,lgraph,options);
%% Save the Network 
save('NN_YOLO.mat');