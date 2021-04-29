cnnFolder = 'C:\Users\ZACHARY JONES\AppData\Local\Temp\';
cnnMatFile = 'imagenet-caffe-alex.mat'; 
cnnFullMatFile = fullfile(cnnFolder, cnnMatFile);
%%Load Pre-trained CNN
% Load MatConvNet network into a SeriesNetwork
convnet = helperImportMatConvNet(cnnFullMatFile);
%%|convnet.Layers| defines the architecture of the CNN 
convnet.Layers