clear; clc; %close all
%cd('C:\Users\mjk10\OneDrive - University of Texas Southwestern\MJKIM\Experiment\fiberPhotometry\hmmData')
% Load your data
data = load('ca20_20240709_1.mat');
timestamps = data.peakVal3(:,1);
filtered_signals = data.peakVal3(30:end-30,3);
filtered_signals = detrend(filtered_signals);
datasize = 20000;
specific_data = zscore(filtered_signals(1:datasize,:));
specific_data = round(specific_data*10000)';


trace1 = data.peakVal3(30:end-30,3);
 [~,valleypos]=findpeaks(-trace1);
 [~,pkpos]=findpeaks(trace1);

 if valleypos(1)>pkpos(1)
     numpos = min(length(valleypos)-1,length(pkpos));
     posdata = [valleypos(2:numpos) pkpos(1:numpos)];
 else
     numpos = min(length(valleypos),length(pkpos));
     posdata = [valleypos(1:numpos) pkpos(1:numpos)];
 end

 maxamp = -Inf;
 for i = 1:size(posdata,1)
     curval = trace1(posdata(i,1):posdata(i,2));
     vl2pkdata{i,1} = curval;
     if abs(curval(end)-curval(1))>maxamp(1)
         maxamp(1) = abs(curval(end)-curval(1));
         maxamp(2) = i;
     end
     clear curval
 end

 % data gathering and recalibration
newarr = zeros(length(trace1),1);
for i = 1:size(posdata,1)-1
    curpos = posdata(i,1):posdata(i,2);
    curpos1 = posdata(i,2):posdata(i+1,1);
    curval = trace1(curpos);
    curval1 = trace1(curpos1);
    curval = curval-curval(1);
    curval1 = curval1-curval1(end);
    curval = curval./maxamp(1);
    curval1 = curval1./curval1(1);
    curval1 = curval1.*curval(end);
    % diffval = curval(end)-curval1(1);
    % if diffval>0
    %     curval1(1:end-1) = curval1(1:end-1)+diffval;
    % else
    %     curval1(1:end-1) = curval1(1:end-1)-diffval;
    % end
    newarr(curpos) = curval;
    newarr(curpos1) = curval1;
    clear cur*
end

% Bin the data
num_bins = 100;
bin_edges = linspace(min(newarr), max(newarr), num_bins + 1);
tmp_signal = zeros(1,length(newarr));
for i = 1:num_bins
    tmp00 = find(bin_edges(i)<newarr & newarr<=bin_edges(i+1));
    tmp_signal(tmp00) = i;
    clear tmp00
end
tmp00 = find(tmp_signal==0);
tmp_signal(tmp00)=1;


% Smoothing with a MA Filter
% windowSize = 500;
% smoothedSignal = smoothdata(specific_data, 'movmean', windowSize);

% % Calculating MSE
% mse = mean((specific_data - smoothedSignal).^2);
% disp(['Mean squared error:' num2str(mse)]);

% Re-scale mapped values from 0 - 100
% min_val = min(smoothedSignal);
% max_val = max(smoothedSignal);,
% baselinePeriod = 1:20000;
% baseline = mean(smoothedSignal(baselinePeriod));`
% recalibratedSignal = smoothedSignal - baseline;
% min_val = min(specific_data);
% max_val = max(specific_data);
% rescaled_signal = (specific_data - min_val) / (max_val - min_val)*100 ;

% Recalibrate
%calibrated_signal = rescaled_states / 200;
% % Bin the data
% num_bins = 100;
% bin_edges = linspace(min(rescaled_signal), max(rescaled_signal), num_bins + 1);
% tmp_signal = zeros(1,length(specific_data));
% for i = 1:num_bins
%     tmp00 = find(bin_edges(i)<rescaled_signal & rescaled_signal<=bin_edges(i+1));
%     tmp_signal(tmp00) = i;
%     clear tmp00
% end
% tmp00 = find(tmp_signal==0);
% tmp_signal(tmp00)=1;
% [~, bins] = histcounts(rescaled_signal, bin_edges);
% bins(bins == 0) = 1; % Assign any zero bins to the first bin



% % Use binned data as states
states = tmp_signal(1:40000);
% uniqueval = unique(states);
% 
% for i = 1:length(un_val)
%     tmp00 =  find(rescaled_signal==un_val(i));
%     states(tmp00) = i;
%     clear tmp00
% end
% 
% 
% % Normalize
% normalized_signals = (filtered_signals - mean(filtered_signals)) / std(filtered_signals);
% 
% % Discretize your data by threshold
% threshold = median(specific_data);
% states = specific_data > threshold; % Binary states: 0 or 1

% Determine number of states (Here the states are unknown, so using BIC)
best_num_states = 1;
best_bic = inf;
for num_states = 1:10 % Guess for No.of Hidden states
    num_states
    % Initialize guesses for transition and emission matrices
    trans_guess = rand(num_states);
    trans_guess = trans_guess ./ sum(trans_guess, 2); % Norm the row to make it 1

    emiss_guess = rand(num_states, num_bins); 
    emiss_guess = emiss_guess ./ sum(emiss_guess, 2);

    % Train the HMM
    [est_trans, est_emiss] = hmmtrain(states, trans_guess, emiss_guess);

    % Compute BIC for the model
    bic = computeBIC(states, est_trans, est_emiss)

    % [~,logL] = hmmdecode(states, est_trans, est_emiss)
    % 
    % num_params = numel(est_trans) + numel(est_emiss);
    % [~,bic1] = aicbic(logL,num_params,datasize)

    % max(bic(:,1))
    if bic < best_bic
        best_bic = bic;
        best_num_states = num_states;
    end

    clear *guess
end

disp(['Best number of states: ', num2str(best_num_states)]);




% % Once the best number of states is found, re-train with that number
trans_guess = rand(best_num_states);
trans_guess = trans_guess ./ sum(trans_guess, 2); % Normalize rows

emiss_guess = rand(best_num_states, num_bins);
emiss_guess = emiss_guess ./ sum(emiss_guess, 2); % Normalize rows

[est_trans, est_emiss] = hmmtrain(states, trans_guess, emiss_guess);


%%
% Decode the hidden states
decoded_states = hmmviterbi(states, est_trans, est_emiss);

% % Display the results
% disp('Estimated Transition Matrix:');
% disp(est_trans);
% disp('Estimated Emission Matrix:');
% disp(est_emiss);
% disp('Decoded States:');
% disp(decoded_states);


% B-I-C

function bic = computeBIC(states, trans, emiss)
% Compute the log-likelihood of the data given the model
[~,logL] = hmmdecode(states, trans, emiss);
% Number of parameters in the model
num_params = numel(trans) + numel(emiss);
% Number of data points
N = length(states);
% Compute BIC
bic = -2 * logL + num_params * log(N);
end