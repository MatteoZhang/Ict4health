clear all
close all
clc 

%% preprocess
rng('default');
tstart=tic;

Kquant=8;% number of quantization levels
Nstates=8;% number of states in the HMM

ktrain=[1,2,3,4,5,6,7     ;
        1,2,3,4,    8,9,10;
            4,5,6,7,8,9,10];% indexes of patients for training
ktest=[ 8,9,10;
        5,6,7;
        1,2,3];% indexes of patients for testing

    n = 1;
ktrain = ktrain(n,:);
ktest = ktest(n,:); 
    
[hq,pq]=pre_process_data(Nstates,Kquant,ktrain);% generate the quantized signals
telapsed = toc(tstart);

disp(['first part, elapsed time ',num2str(telapsed),' s'])



%% HMM training phase....

rng(1) %random seed
transition_matrix_hat = rand(Nstates, Nstates); 
emission_matrix_hat =rand(Nstates, Kquant);

%normalize sum of each row should be equal to 1
tot = sum(transition_matrix_hat,2);
tot2 = sum(emission_matrix_hat,2);

for i = 1:Nstates
    transition_matrix_hat(i,:) = transition_matrix_hat(i,:)/tot(i);
    emission_matrix_hat(i,:) = emission_matrix_hat(i,:)/tot2(i);
end

tstart1=tic;
% healty patient state machine training 
[TR_H,EMIT_H]=hmmtrain(hq(ktrain),transition_matrix_hat,emission_matrix_hat,'Tolerance',1e-3,'Maxiterations',200);
% parkinson patient state machine training
[TR_P,EMIT_P]=hmmtrain(pq(ktrain),transition_matrix_hat,emission_matrix_hat,'Tolerance',1e-3,'Maxiterations',200);

%% HMM testing phase....

%sensitivity = true positive/(true positive + false negatives)
% prob of positive test given disease
%specificity = true negative/(true negative + false positive)
% negative test given well

train_specificity = 0;
for i = ktrain
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H > logp_P
        train_specificity = train_specificity + 1/length(ktrain);
    end
end

train_sensitivity = 0;
for i = ktrain()
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H < logp_P
        train_sensitivity = train_sensitivity + 1/length(ktrain);
    end
end

test_specificity = 0;
for i = ktest
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H > logp_P
        test_specificity = test_specificity + 1/length(ktest);
    end
end

test_sensitivity = 0;
for i = ktest
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H < logp_P
        test_sensitivity = test_sensitivity + 1/length(ktest);
    end
end

telapsed1=toc(tstart1);

clear transition_matrix_hat

% Start second part with cirlulant matrix

p = 0.9;
q = (1 - p) / (Nstates - 1);
qs = q*ones(1, Nstates-1);
v = [p qs];
transition_matrix_hat = zeros(Nstates,Nstates);
for i = 1:Nstates
    
   transition_matrix_hat(i,:) =  circshift(v,1);
   v = transition_matrix_hat(i,:);
end

tstart2=tic;
% healty patient state machine training 
[TR_H,EMIT_H]=hmmtrain(hq(ktrain),transition_matrix_hat,emission_matrix_hat,'Tolerance',1e-3,'Maxiterations',200);
% parkinson patient state machine training
[TR_P,EMIT_P]=hmmtrain(pq(ktrain),transition_matrix_hat,emission_matrix_hat,'Tolerance',1e-3,'Maxiterations',200);

pqtrain_specificity = 0;
for i = ktrain
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H > logp_P
        pqtrain_specificity = pqtrain_specificity + 1/length(ktrain);
    end
end

pqtrain_sensitivity = 0;
for i = ktrain()
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H < logp_P
        pqtrain_sensitivity = pqtrain_sensitivity + 1/length(ktrain);
    end
end

pqtest_specificity = 0;
for i = ktest
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H > logp_P
        pqtest_specificity = pqtest_specificity + 1/length(ktest);
    end
end

pqtest_sensitivity = 0;
for i = ktest
    [~, logp_H] = hmmdecode(hq{i}, TR_H,EMIT_H);
    [~, logp_P] = hmmdecode(pq{i}, TR_P,EMIT_P);
    if logp_H < logp_P
        pqtest_sensitivity = pqtest_sensitivity + 1/length(ktest);
    end
end

telapsed2=toc(tstart2);


clc

ktrain
ktest

res1 = [train_sensitivity train_specificity; test_sensitivity test_specificity]
res2 = [pqtrain_sensitivity pqtrain_specificity; pqtest_sensitivity pqtest_specificity]

disp(['Time Random init ', num2str(telapsed1+telapsed), ' s'])
disp(['Time Circulant init ', num2str(telapsed2+telapsed), ' s'])
