%% Initialize error results matrices
errSq = zeros(numStartPoints, numPredSteps, numRepeat); % Squares of 
% distance-error for numPredSteps predicting-steps at numStartPoints 
% start-points with numRepeat repeated trials.
errSqMeanPred = zeros(numStartPoints,numPredSteps);% Squares of 
% distance-error with the mean of past observations for prediciton (non repeats since no randomness)
errSqLatestPred = zeros(numStartPoints,numPredSteps);% Squares of 
% distance-error with the last observation for prediction (non repeats since no randomness)
rmseTraining = zeros(numRepeat,numStartPoints,maxEpochs); % single point's RMSE 
% at each epoch for numRepeat*numStartPoints training sessions 
%% Define input and output of prediction model (the same setting used for all models)
numInputDim = size(data,1);
numResponses = 3*outputSteps;
numHiddenUnits = 300; % the number of the hidden units of LSTM module 
%% Start numStartPoints-step prediction at various starting points for multiple repeated tests
for iter = 1:numRepeat
    % re-initialize the same net for each test
    layers = [ ...
        sequenceInputLayer(numInputDim)
        lstmLayer(numHiddenUnits)
        fullyConnectedLayer(numResponses)
        regressionLayer];
    % use the same options for training
    options = trainingOptions('adam', 'MaxEpochs',maxEpochs,'GradientThreshold',1, ...
        'ExecutionEnvironment','gpu', 'InitialLearnRate',0.01, 'Verbose',1,...
        'LearnRateSchedule','piecewise','LearnRateDropPeriod',100,'LearnRateDropFactor',0.2);
    for sp = 1:numStartPoints
        startPoint = 599+sp;  % set the start-point of prediction
        endTrain = startPoint-outputSteps;   % set the end-point of the inputted training sequence
       %% Standardize the training data {XTrain, YTrain}
        mu = mean(data(:,1:startPoint),2);
        sig = std(data(:,1:startPoint),0,2);
        dataStd = (data - mu)./sig;
        XTrain = dataStd(:,1:endTrain); 
        YTrain = zeros(numResponses,endTrain);
        for ii=0:outputSteps-1
            YTrain(ii*3+(1:3),:) = dataStd(end-3+1:end,ii+(2:endTrain+1));
        end
       %% Train the net at current start-point
        [net, info]= trainNetwork(XTrain,YTrain,layers,options);
        rmseTraining(iter, sp,:) = (info.TrainingRMSE)/sqrt(outputSteps); % save the RMSE for single point at each epoch
       %% Cycle prediction with the trained net at current start-point
        YPred = cyclePredictWithRNN(dataStd, startPoint, numPredSteps, net);
       %% Compute prediction errors
        YTest = dataStd(end-2:end,startPoint+(1:numPredSteps)); % True output sequence
        errSq(sp,:,iter) = sum((sig(end-2:end).*(YPred-YTest)).^2); % save distance-error to errSq matrix
       %% Compute the reference errors
        if iter == 1
            errSqMeanPred(sp,:) = sum((YTest.*sig(end-3+1:end)).^2); % for mean prediction
            errSqLatestPred(sp,:) = sum((sig(end-3+1:end).*(dataStd(end-3+1:end,startPoint)-YTest)).^2); % for no prediction i.e. use the last observation as predicted value
        end
        disp([num2str(iter), ' - ',num2str(startPoint) ,' finished!',...
            ' trainRmse = ' num2str(rmseTraining(iter, sp,end)),...
            ' totalRmse = ' num2str(sqrt(mean(errSq(sp,:,iter)))),...
            ' meanRefRmse = ' num2str(sqrt(mean(sum((YTest.*sig(end-3+1:end)).^2))))]) % display prompts during code running 
    end 
end  
 