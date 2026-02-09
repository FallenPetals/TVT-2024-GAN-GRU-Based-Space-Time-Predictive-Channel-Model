% load('APDP_LOS_24G_new2.mat'); % Load the channel data

CIR = 20*log10(abs((APDP(51:350,:,1))));
save('LOS_meas.mat', 'CIR'); % Save result to file

Len = size(CIR, 1);  
NumPositions = size(CIR, 2); 

W = 30; % Width of delay segmentation
ovlp = 10; % Overlap width
T1 = -70; % Threshold for peak value of LoS path (adjustable)
T2 = -85; % Threshold for average value of LoS path (adjustable)

Result = strings(Len, NumPositions); % Initialize result array

for pos = 1:NumPositions
    S_qp_fc = (CIR(:, pos)); % Compute delay PSD
    
    for i = 1:(W - ovlp):(Len - W + 1)
        S_i = S_qp_fc(i:i+W-1); % Extract segment
        
        S_peak = max(S_i);
        S_mean = mean(S_i);
        
        if (S_peak > T1) && (S_mean > T2)
            Result(i:i+W-1, pos) = "LoS";
        else
            Result(i:i+W-1, pos) = "NLoS";
        end
    end
end

% Post-processing to correct misclassified segments
for pos = 1:NumPositions
    for i = 1:Len
        if i > 1 && i < Len
            if strcmp(Result(i-1, pos), "LoS") && strcmp(Result(i+1, pos), "LoS")
                Result(i, pos) = "LoS"; % Correct isolated NLoS segments
            end
        end
    end
end