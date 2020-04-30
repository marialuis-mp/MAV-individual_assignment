function [TP,FP,FN] = Calculate_TP_FP_FN(IoU, J, K, Threshold)
    TP1=0;
    
    % Calculate the number of True Positives
    Max_gate = max(IoU,[],2);
    for j = 1:J
        if Max_gate(j) > Threshold
            TP1 = TP1+1;
        end
    end
    TP = min(TP1,K);
    FP = K - TP;
    FN = max(J - K, 0);
end

