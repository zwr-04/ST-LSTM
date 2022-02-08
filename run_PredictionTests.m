clc; close all; clear all;
for testId = 1:10
    %% load data
    if testId<=5
        load measuresInvivo.mat measures % load in vivo measures
    else
        load measuresPhantom.mat measures % load phantom measures
    end
    % Notes: measures is a 12x750 matrix, containing 3D coordinates of 4
    % points (one POI and three auxiliary points (AP)) on the same heart
    % surfaces at 750 sampling times. The column is stacked in the order of
    % [POI; AP1; AP2; AP3]
    %% set parameters for differet testing
    switch testId
        %-----------------------in vivo data-------------------------------
        case 1 % test 1-step LSTM on in vivo data
            outputSteps = 1;    % for single-step prediction
            maxEpochs = 100; % set the max epochs for training
            data = measures(1:3,:); % extract the POI measures
        case 2 % test 10-step LSTM on in vivo data
            outputSteps = 10;    % for 10-step prediction
            maxEpochs = 200; 
            data = measures(1:3,:);
        case 3 % test 20-step LSTM on in vivo data
            outputSteps = 20;    % for 20-step prediction
            maxEpochs = 150;
            data = measures(1:3,:);
        case 4 % test 40-step LSTM on in vivo data
            outputSteps = 40;    % for 40-step prediction
            maxEpochs = 50;
            data = measures(1:3,:);
        case 5 % test ST-LSTM on in vivo data
            outputSteps = 1;    % output steps of the model for single prediction
            maxEpochs = 50;
            data = measures; % all 4 points are used for training
        %-----------------------phantom data ------------------------------
        case 6 % test 1-step LSTM on phantom data
            outputSteps = 1;    
            maxEpochs = 350; 
            data = measures(1:3,:); 
        case 7 % test 10-step LSTM on phantom data
            outputSteps = 10;
            maxEpochs = 400;
            data = measures(1:3,:); 
        case 8 % test 20-step LSTM on phantom data
            outputSteps = 20;
            maxEpochs = 500;
            data = measures(1:3,:); 
        case 9 % test 40-step LSTM on phantom data
            outputSteps = 40;
            maxEpochs = 500; 
            data = measures(1:3,:); 
        case 10 % test ST-LSTM on phantom data
            outputSteps = 1;  
            maxEpochs = 300;
            data = measures;
    end

    %% Setting task parameters for all tests
    numPredSteps=100;   % future steps for prediction
    numRepeat = 2;     % repeat times for each starting point of prediction
    numStartPoints = 2;  % number of starting points for prediction test
      
    %% Run prediction tests and save results
    LSTM_Prediction % run LSTM based prediction
    save(['lstmTest_' num2str(testId) '_' num2str(numRepeat) 'x' num2str(numStartPoints) '.mat'], 'errSq', 'errSqMeanPred', 'errSqLatestPred', 'rmseTraining', 'net')
end