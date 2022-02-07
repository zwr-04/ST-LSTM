function [YPred, YPredPre]= cyclePredictWithRNN(dataStd, startPoint, numPredSteps, net)
    inputDim = size(dataStd,1);
    outputDim = net.Layers(end-1).OutputSize;
    XTestPre = dataStd(:,1:startPoint-1);
    net = resetState(net);
    [net,YPredPre] = predictAndUpdateState(net,XTestPre); % update net to startPoint-1
    YPredPre = YPredPre(1:3,:);
    xTest = dataStd(:,startPoint);
    if inputDim==3 % prediction with temporal correlation LSTM
        outputSteps = outputDim/inputDim;
        YPred = zeros(inputDim,ceil(numPredSteps/outputSteps)*outputSteps);
        for cyc = 1:ceil(numPredSteps/outputSteps)
            [net, yPred] = predictAndUpdateState(net,xTest); % output the prediction results at xTest
            xTest = reshape(yPred(:,end),[],outputSteps); % setting prediction results at the last point as the input of the next predicition cycle 
            YPred(:,(cyc-1)*outputSteps+(1:outputSteps)) = xTest; % saving the prediction results at current cycle
%             if cyc == 50
%                 xTest(:,end) = xTest(:,end)-10;
%             end
        end
        YPred = YPred(:,1:numPredSteps);
    else % prediction with spatial-temporal correlation LSTM
        YPred = zeros(outputDim,numPredSteps);
        for cyc = 1:numPredSteps
            [net, YPred(:,cyc)] = predictAndUpdateState(net,xTest); % output the prediction results at xTest
%             if cyc == 50
%                 xTest = [dataStd(1:end-outputDim,startPoint+cyc); YPred(:,cyc)-10];
%             else
                xTest = [dataStd(1:end-outputDim,startPoint+cyc); YPred(:,cyc)];
%             end
        end
    end  
end