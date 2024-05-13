data_dir = 'C:/Users/48512/Desktop/Maleria_Matlab';
train_dir = fullfile(data_dir, 'cell_images');

img_size = [28,28];

normal_cases_dir = fullfile(train_dir, 'Uninfected');
malaria_cases_dir = fullfile(train_dir, 'Parasitized');

normal_cases = dir(fullfile(normal_cases_dir, '*.png'));
malaria_cases = dir(fullfile(malaria_cases_dir, '*.png'));

%% 

train_data = [];
for i = 1:numel(normal_cases)
    train_data = [train_data; {fullfile(normal_cases_dir, normal_cases(i).name)}, 0];
end

for i = 1:numel(malaria_cases)
    train_data = [train_data; {fullfile(malaria_cases_dir, malaria_cases(i).name)}, 1];
end

%% 

 
    train_data_table = cell2table(train_data, 'VariableNames', {'image', 'label'});
   
%% 


cases_count = countcats(categorical(train_data_table.label));
disp(cases_count);

figure('Position', [100, 100, 800, 600]);
bar(cases_count);
title('Number of cases');
xlabel('Case type');
ylabel('Count');
xticklabels({'Uninfected(0)', 'Parasitized(1)'});
%% 

malaria_samples_indices = find(train_data_table.label == 1);
normal_samples_indices = find(train_data_table.label == 0);

malaria_samples = table2array(train_data_table(malaria_samples_indices(1:5), 'image'));
normal_samples = table2array(train_data_table(normal_samples_indices(1:5), 'image'));


samples = [malaria_samples, normal_samples];

%% 

figure('Position', [100, 100, 1800, 600]);
for i = 1:numel(samples)
    subplot(2, 5, i);
    img = imread(samples{i});
    imshow(img);
    if i <= 5
        title('Malaria Samples', 'FontSize', 14);
    else
        title('Normal Samples', 'FontSize', 14);
    end
    axis off;
end
%% 


traing_data = [];
train_labels = [];

disp('Processing normal cases...');
for i = 1:numel(normal_cases)
    img = imread(fullfile(normal_cases_dir, normal_cases(i).name));
    img = imresize(img, img_size);
    if size(img, 3) == 1
        img = cat(3, img, img, img); 
    end
    img = im2double(img); 
    label = 'normal';
    traing_data{end+1} = img;
    train_labels{end+1} = label;
end
%% 

disp('Processing malaria cases...');
for i = 1:numel(malaria_cases)
    img = imread(fullfile(malaria_cases_dir, malaria_cases(i).name));
    img = imresize(img, img_size);
    if size(img, 3) == 1
        img = cat(3, img, img, img); 
    end
    img = im2double(img); 
    label = 'malaria';
    traing_data{end+1} = img;
    train_labels{end+1} = label;
end
%% 

train_data1 = traing_data';
train_labels1 = train_labels;


disp(['Total number of validation examples: ', num2str(size(train_data1, 4))]);
disp(['Total number of labels: ', num2str(numel(train_labels1))]);
%% 
train_labels1_categorical = categorical(train_labels1);

train_labels1_table = table(train_labels1_categorical, 'VariableNames', {'label'});
%% 

train_labels1_num = grp2idx(train_labels1) - 1; 

%% 
data = table(train_data1, train_labels1_categorical');


%% 

train_labels1 = categorical(train_labels1);


rng(42); 
numRows = size(data, 1);
percentTraining = 0.7;
numTraining = round(numRows * percentTraining);

idx = randperm(numRows);
idxTraining = idx(1:numTraining);
idxTesting = idx(numTraining+1:end);
dataTraining = data(idxTraining, :);
dataTesting = data(idxTesting, :);

% %% 
%% 
% 
num_images = numel(train_data1);
num_labels = numel(train_labels1); 

disp(num_images);
disp(num_labels);




%% 
layers = [
imageInputLayer([img_size 3]),
convolution2dLayer(3,64,'Padding','same'),
batchNormalizationLayer,
reluLayer,
maxPooling2dLayer(2,'Stride',2),

convolution2dLayer(5,128,'Padding','same'),
batchNormalizationLayer,
reluLayer,
maxPooling2dLayer(2,'Stride',2),

convolution2dLayer(5,256,'Padding','same'),
batchNormalizationLayer,
reluLayer,
maxPooling2dLayer(2,'Stride',2),

convolution2dLayer(5,256,'Padding','same'),
batchNormalizationLayer,
reluLayer,
maxPooling2dLayer(2,'Stride',2),

fullyConnectedLayer(2),
softmaxLayer,
classificationLayer
];

% Define options for training
options = trainingOptions("adam", ...
    MaxEpochs=5, ...
    InitialLearnRate=0.01,...
    Shuffle="every-epoch", ...
    GradientThreshold=1, ...
    Verbose=false, ...
    Plots="training-progress");
%% 



net = trainNetwork(dataTraining, layers, options);
%% 
save('trained_model.mat', 'net');
%% 

loaded_model = load('trained_model.mat');

net = loaded_model.net;



%% 
 
test_img = imread('cell_images\Parasitized\C33P1thinF_IMG_20150619_114756a_cell_179.png');

test_img = imresize(test_img, [28,28]);

predictions = classify(net, test_img);


 disp(predictions);

