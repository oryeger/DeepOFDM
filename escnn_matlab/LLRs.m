%% llr_histogram_64qam_multipeak.m
clear; clc; close all;

%% Parameters
M = 64;
k = log2(M);
Nsym = 1e6;
snr_dB = 22;          % increase SNR to reveal peaks
numBins = 400;

%% Generate bits and symbols
txBits = randi([0 1], Nsym, k);
txIdx  = bi2de(txBits, 'left-msb');

txSym = qammod(txIdx, M, ...
    'gray', ...
    'InputType', 'integer', ...
    'UnitAveragePower', true);

%% AWGN
snrLin   = 10^(snr_dB/10);
noiseVar = 1 / snrLin;
noise = sqrt(noiseVar/2) * (randn(size(txSym)) + 1i*randn(size(txSym)));
rxSym = txSym + noise;

%% Exact LLRs
llr = qamdemod(rxSym, M, ...
    'gray', ...
    'OutputType', 'llr', ...
    'UnitAveragePower', true, ...
    'NoiseVariance', noiseVar);

%% Make sure llr is Nsym x 6
if isvector(llr)
    llr = reshape(llr, Nsym, k);
elseif size(llr,1) == k && size(llr,2) == Nsym
    llr = llr.';
end

%% Plot: one subplot per bit, conditioned on transmitted bit
figure('Color','w','Position',[100 100 1200 700]);

for b = 1:k
    subplot(2,3,b); hold on;

    histogram(llr(txBits(:,b)==0,b), numBins, ...
        'Normalization','pdf', ...
        'DisplayStyle','stairs', ...
        'LineWidth',1.5);

    histogram(llr(txBits(:,b)==1,b), numBins, ...
        'Normalization','pdf', ...
        'DisplayStyle','stairs', ...
        'LineWidth',1.5);

    grid on;
    xlabel('LLR');
    ylabel('PDF');
    title(sprintf('Bit %d', b));
    legend('tx bit = 0','tx bit = 1','Location','best');
end

sgtitle(sprintf('64-QAM exact bit LLR histograms, conditioned on transmitted bit, SNR = %.1f dB', snr_dB));