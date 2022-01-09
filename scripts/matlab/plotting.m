%% PLOTTING RESULTS FROM TRAINING OF AGENT progress.csv %%

% Loading the experiment results
experiment1 = 'pickplacetarget_20.07.21_50demofixed_rnd';
experiment2 = 'pickplacetarget_20.07.21_50demo_rnd1';
experiment3 = 'pickplacetarget_20.07.21_50demo_rnd2';
experiment4 = 'pickplacetarget_09.07.21_50demo_rnd';
experiment5 = 'pickplacernet_12.07.21_50demo_fixed';
experiment6 = 'pickplacernet_12.07.21_50demo_rnd';
exp1 = readtable(strcat(experiment1, '/', 'progress.csv'));
exp2 = readtable(strcat(experiment2, '/', 'progress.csv'));
exp3 = readtable(strcat(experiment3, '/', 'progress.csv'));
exp4 = readtable(strcat(experiment4, '/', 'progress.csv'));
exp5 = readtable(strcat(experiment5, '/', 'progress.csv'));
exp6 = readtable(strcat(experiment6, '/', 'progress.csv'));

% Loading the experiment results using eval env
experimentA1 = 'pickplacetarget_13.09.21_50demo_fix_rnd';
experimentA2 = 'pickplacetarget_13.09.21_50demo_fix';
experimentB = 'pickplacetarget_07.09.21_50demo_rnd';
experimentC1 = 'pickplacetarget_05.09.21_50demo_rnd';
experimentC2 = 'pickplacetarget_08.09.21_50demo_rnd';
experimentD = 'pickplacetarget_11.09.21_48demo_rnd';
experimentE = 'pickplacetarget_10.09.21_50demo_fixed';
expA1 = readtable(strcat(experimentA1, '/', 'progress.csv'));
expA2 = readtable(strcat(experimentA2, '/', 'progress.csv'));
expB = readtable(strcat(experimentB, '/', 'progress.csv'));
expC1 = readtable(strcat(experimentC1, '/', 'progress.csv'));
expC2 = readtable(strcat(experimentC2, '/', 'progress.csv'));
expD = readtable(strcat(experimentD, '/', 'progress.csv'));
expE = readtable(strcat(experimentE, '/', 'progress.csv'));


%%
x0 = 10;
y0 = 10;
width = 1000;
height = 500;


%% Plotting training and testaing success rate
fig = figure(3)
hold on
subplot(2,2,1)
sgtitle('\fontsize{40}Reach kidney task train for 400K steps')
plot(exp1.epoch, exp1.train_success_rate, 'Color', [1, 0, 0]);
hold on
plot(exp2.epoch, exp2.train_success_rate, 'Color', [1, 0.563, 0]);
hold on
plot(exp3.epoch, exp3.train_success_rate, 'Color', [0.5, 0, 1]);
hold on
plot(exp4.epoch, exp4.train_success_rate, 'Color', [1, 0, 0.856]);
hold on
plot(exp5.epoch, exp5.train_success_rate, 'Color', [0, 1, 0.8]);
hold on
plot(exp6.epoch, exp6.train_success_rate, 'Color', [1, 0.8, 0]);

%legend('fixed demo rnd', 'rnd1', 'rnd2', 'rnd grasp', 'rnd grasp', 'all rnd', 'Location', 'SouthEast')



title("Training Succes Rate")
xlabel("Epoch")
ylabel("Train success rate")
%ylim([0 100])
set(gca,'FontSize',25)
grid on
hold on


figure(3)
hold on
subplot(2,2,2)
%plot(exp.epoch, exp.test_success_rate, 'Color', 'r');
%plot(exp.epoch, exp.test_success_rate);
plot(exp1.epoch, exp1.test_success_rate, 'Color', [1, 0, 0]);
hold on
plot(exp2.epoch, exp2.test_success_rate, 'Color', [1, 0.563, 0]);
hold on
plot(exp3.epoch, exp3.test_success_rate, 'Color', [0.5, 0, 1]);
hold on
plot(exp4.epoch, exp4.test_success_rate, 'Color', [1, 0, 0.856]);
hold on
plot(exp5.epoch, exp5.test_success_rate, 'Color', [0, 1, 0.8]);
hold on
plot(exp6.epoch, exp6.test_success_rate, 'Color', [1, 0.8, 0]);


title("Test Succes Rate")
xlabel("Epoch")
ylabel("Test success rate")
%ylim([0 100])
set(gca,'FontSize',25)
grid on

% Plotting evaluations TRAIN
figure(3)
hold on
subplot(2,2,3)
%plot(exp.epoch, exp.test_success_rate, 'Color', 'r');
%plot(exp.epoch, exp.test_success_rate);
plot(expA1.epoch, expA1.train_success_rate, 'Color', [1, 0, 0]);
hold on
plot(expA2.epoch, expA2.train_success_rate, 'Color', [1, 0.563, 0]);
hold on
plot(expB.epoch, expB.train_success_rate, 'Color', [0.5, 0, 1]);
hold on
plot(expC1.epoch, expC1.train_success_rate, 'Color', [1, 0, 0.856]);
hold on
plot(expC2.epoch, expC2.train_success_rate, 'Color', [0, 1, 0.8]);
hold on
plot(expD.epoch, expD.test_success_rate, 'Color', [1, 0.8, 0]);
hold on
plot(expE.epoch, expE.test_success_rate, 'Color', [1, 0.2, 0]);


title("Test Succes Rate")
xlabel("Epoch")
ylabel("Test success rate")
%ylim([0 100])
set(gca,'FontSize',25)
grid on


% Plotting evaluations TEST
figure(3)
hold on
subplot(2,2,4)
%plot(exp.epoch, exp.test_success_rate, 'Color', 'r');
%plot(exp.epoch, exp.test_success_rate);
plot(expA1.epoch, expA1.test_success_rate, 'Color', [1, 0, 0]);
hold on
plot(expA2.epoch, expA2.test_success_rate, 'Color', [1, 0.563, 0]);
hold on
plot(expB.epoch, expB.test_success_rate, 'Color', [0.5, 0, 1]);
hold on
plot(expC1.epoch, expC1.test_success_rate, 'Color', [1, 0, 0.856]);
hold on
plot(expC2.epoch, expC2.test_success_rate, 'Color', [0, 1, 0.8]);
hold on
plot(expD.epoch, expD.test_success_rate, 'Color', [1, 0.8, 0]);
hold on
plot(expE.epoch, expE.test_success_rate, 'Color', [1, 0.2, 0]);


title("Test Succes Rate")
xlabel("Epoch")
ylabel("Test success rate")
%ylim([0 100])
set(gca,'FontSize',25)
grid on


%%
%success_rate_avg= smooth(exp.train_success_rate);

figure(3)
hold on
grid on 
rd = 5;
fl = 21;
smoth_training1  = sgolayfilt(exp1.train_success_rate,rd,fl);
smoth_training2  = sgolayfilt(exp2.train_success_rate,rd,fl);
smoth_training3  = sgolayfilt(exp3.train_success_rate,rd,fl);
smoth_training4  = sgolayfilt(exp4.train_success_rate,rd,fl);
smoth_training5  = sgolayfilt(exp5.train_success_rate,rd,fl);
smoth_training6  = sgolayfilt(exp6.train_success_rate,rd,fl);

subplot(2,2,1)
hold on
plot(exp1.epoch, smoth_training1, 'Color', [1, 0, 0], 'LineWidth', 4);
hold on
plot(exp2.epoch, smoth_training2, 'Color', [1, 0.563, 0], 'LineWidth', 4);
hold on
plot(exp3.epoch, smoth_training3, 'Color', [0.5, 0, 1], 'LineWidth', 4);
hold on
plot(exp4.epoch, smoth_training4, 'Color', [1, 0, 0.856], 'LineWidth', 4);
hold on
plot(exp5.epoch, smoth_training5, 'Color', [0, 1, 0.8], 'LineWidth', 4);
hold on
plot(exp6.epoch, smoth_training6, 'Color', [1, 0.8, 0], 'LineWidth', 4);

%plot(exp.epoch, smoth_training, 'Color', 'r', 'LineWidth', 4);
%plot(exp.epoch, success_rate_avg, 'Color', 'b', 'LineWidth', 4);


smoth_test1 = sgolayfilt(exp1.test_success_rate,rd,fl);
smoth_test2 = sgolayfilt(exp2.test_success_rate,rd,fl);
smoth_test3 = sgolayfilt(exp3.test_success_rate,rd,fl);
smoth_test4 = sgolayfilt(exp4.test_success_rate,rd,fl);
smoth_test5 = sgolayfilt(exp5.test_success_rate,rd,fl);
smoth_test6 = sgolayfilt(exp6.test_success_rate,rd,fl);

%legend('fixed demo rnd', 'rnd1', 'rnd2', 'rnd grasp', 'rnd grasp', 'all rnd', 'Location', 'SouthEast')
legend('rnd2 eval', 'rnd2')

subplot(2,2,2)
hold on
plot(exp1.epoch, smoth_test1, 'Color', [1, 0, 0], 'LineWidth', 4);
hold on
plot(exp2.epoch, smoth_test2, 'Color', [1, 0.563, 0], 'LineWidth', 4);
hold on
plot(exp3.epoch, smoth_test3, 'Color', [0.5, 0, 1], 'LineWidth', 4);
hold on
plot(exp4.epoch, smoth_test4, 'Color', [1, 0, 0.856], 'LineWidth', 4);
hold on
plot(exp5.epoch, smoth_test5, 'Color', [0, 1, 0.8], 'LineWidth', 4);
hold on
plot(exp6.epoch, smoth_test6, 'Color', [1, 0.8, 0], 'LineWidth', 4);


%plot(exp.epoch, smoth_test, 'Color', 'r', 'LineWidth', 4);


% EVAL 
figure(3)
hold on
grid on 
rd = 5;
fl = 21;
smoth_trainingA1  = sgolayfilt(expA1.train_success_rate,rd,fl);
smoth_trainingA2  = sgolayfilt(expA2.train_success_rate,rd,fl);
smoth_trainingB  = sgolayfilt(expB.train_success_rate,rd,fl);
smoth_trainingC1  = sgolayfilt(expC1.train_success_rate,rd,fl);
smoth_trainingC2  = sgolayfilt(expC2.train_success_rate,rd,fl);
smoth_trainingD  = sgolayfilt(expD.train_success_rate,rd,fl);
smoth_trainingE  = sgolayfilt(expE.train_success_rate,rd,fl)

subplot(2,2,3)
hold on
plot(expA1.epoch, smoth_trainingA1, 'Color', [1, 0, 0], 'LineWidth', 4);
hold on
plot(expA2.epoch, smoth_trainingA2, 'Color', [1, 0.563, 0], 'LineWidth', 4);
hold on
plot(expB.epoch, smoth_trainingB, 'Color', [0.5, 0, 1], 'LineWidth', 4);
hold on
plot(expC1.epoch, smoth_trainingC1, 'Color', [1, 0, 0.856], 'LineWidth', 4);
hold on
plot(expC2.epoch, smoth_trainingC2, 'Color', [0, 1, 0.8], 'LineWidth', 4);
hold on
plot(expD.epoch, smoth_trainingD, 'Color', [1, 0.8, 0], 'LineWidth', 4);
hold on
plot(expE.epoch, smoth_trainingE, 'Color', [1, 0.2, 0], 'LineWidth', 4);


smoth_testA1  = sgolayfilt(expA1.test_success_rate,rd,fl);
smoth_testA2  = sgolayfilt(expA2.test_success_rate,rd,fl);
smoth_testB  = sgolayfilt(expB.test_success_rate,rd,fl);
smoth_testC1  = sgolayfilt(expC1.test_success_rate,rd,fl);
smoth_testC2  = sgolayfilt(expC2.test_success_rate,rd,fl);
smoth_testD  = sgolayfilt(expD.test_success_rate,rd,fl);
smoth_testE  = sgolayfilt(expE.test_success_rate,rd,fl);

subplot(2,2,4)
hold on
plot(expA1.epoch, smoth_testA1, 'Color', [1, 0, 0], 'LineWidth', 4);
hold on
plot(expA2.epoch, smoth_testA2, 'Color', [1, 0.563, 0], 'LineWidth', 4);
hold on
plot(expB.epoch, smoth_testB, 'Color', [0.5, 0, 1], 'LineWidth', 4);
hold on
plot(expC1.epoch, smoth_testC1, 'Color', [1, 0, 0.856], 'LineWidth', 4);
hold on
plot(expC2.epoch, smoth_testC2, 'Color', [0, 1, 0.8], 'LineWidth', 4);
hold on
plot(expD.epoch, smoth_testD, 'Color', [1, 0.8, 0], 'LineWidth', 4);
hold on
plot(expE.epoch, smoth_testE, 'Color', [1, 0.2, 0], 'LineWidth', 4);



%% Plotting success rate all together
figure(4)
subplot(1,2,1)
hold on 
%plot(exp.epoch, smoth_training, 'Color',[0.4660, 0.6740, 0.1880] , 'LineWidth', 4);
%plot(exp.epoch, smoth_training, 'Color',[0.8660, 0.2740, 0.0880] , 'LineWidth', 4);
%plot(exp.epoch, smoth_training, 'LineWidth', 4);

plot(exp1.epoch, smoth_training1, 'Color', [1, 0, 0], 'LineWidth', 4);
hold on
plot(exp2.epoch, smoth_training2, 'Color', [1, 0.884, 0], 'LineWidth', 4);
hold on
plot(exp3.epoch, smoth_training3, 'Color', [0.5, 0, 1], 'LineWidth', 4);
hold on
plot(exp4.epoch, smoth_training4, 'Color', [1, 0, 0.856], 'LineWidth', 4);
hold on
plot(exp5.epoch, smoth_training5, 'Color', [0, 1, 0.8], 'LineWidth', 4);
hold on
plot(exp6.epoch, smoth_training6, 'Color', [1, 0.8, 0], 'LineWidth', 4);



grid on
hold on
%plot(exp.epoch, smoth_test, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 4);
%plot(exp.epoch, smoth_test, 'Color', [0.6, 0.2470, 0.3410], 'LineWidth', 4);
%plot(exp.epoch, smoth_test, 'LineWidth', 4);

plot(exp1.epoch, smoth_test1, 'Color', [1, 0.469, 0], 'LineWidth', 4);
hold on
plot(exp2.epoch, smoth_test2, 'Color', [0.969, 1, 0], 'LineWidth', 4);
hold on
plot(exp3.epoch, smoth_test3, 'Color', [0.2, 0, 1], 'LineWidth', 4);
hold on
plot(exp4.epoch, smoth_test4, 'Color', [1, 0, 0.496], 'LineWidth', 4);
hold on
plot(exp5.epoch, smoth_test5, 'Color', [0, 1, 0.4], 'LineWidth', 4);
hold on
plot(exp6.epoch, smoth_test6, 'Color', [1, 0.4, 0], 'LineWidth', 4);



title("Success Rates - Pick place target 400k steps using 50 demonstrations")
xlabel("Epoch", 'FontSize',35)
ylabel("Success rate")
set(gca,'FontSize',35)
%legend('Test Success Rate - FIXED','Train Success Rate - FIXED','FontSize',35)
hold on 
%legend('Test Success Rate - RND','Train Success Rate - RND','FontSize',35)

%legend('rnd2 evalTR', 'rnd2-TR', 'rnd2 evalTS', 'rnd2-TS')
%legend('fixed demo rnd-TR', 'rnd1-TR', 'rnd2-TR', 'rnd grasp-TR', 'rnd grasp-TR', 'all rnd-TR', 'fixed demo rnd-TS', 'rnd1-TS', 'rnd2-TS', 'rnd grasp-TS', 'rnd grasp-TS', 'all rnd-TS', 'Location', 'SouthEast')

figure(4)
subplot(1,2,2)
hold on 
plot(expA1.epoch, smoth_trainingA1, 'Color', [1, 0, 0], 'LineWidth', 4);
hold on
plot(expA2.epoch, smoth_trainingA2, 'Color', [1, 0.884, 0], 'LineWidth', 4);
hold on
plot(expB.epoch, smoth_trainingB, 'Color', [0.5, 0, 1], 'LineWidth', 4);
hold on
plot(expC1.epoch, smoth_trainingC1, 'Color', [1, 0, 0.856], 'LineWidth', 4);
hold on
plot(expC2.epoch, smoth_trainingC2, 'Color', [0, 1, 0.8], 'LineWidth', 4);
hold on
plot(expD.epoch, smoth_trainingD, 'Color', [1, 0.8, 0], 'LineWidth', 4);
hold on
plot(expE.epoch, smoth_trainingE, 'Color', [1, 0.2, 0], 'LineWidth', 4);


grid on 
hold on 
plot(expA1.epoch, smoth_testA1, 'Color', [1, 0.469, 0], 'LineWidth', 4);
hold on
plot(expA2.epoch, smoth_testA2, 'Color', [0.969, 1, 0], 'LineWidth', 4);
hold on
plot(expB.epoch, smoth_testB, 'Color', [0.2, 0, 1], 'LineWidth', 4);
hold on
plot(expC1.epoch, smoth_testC1, 'Color', [1, 0, 0.496], 'LineWidth', 4);
hold on
plot(expC2.epoch, smoth_testC2, 'Color', [0, 1, 0.4], 'LineWidth', 4);
hold on
plot(expD.epoch, smoth_testD, 'Color', [1, 0.4, 0], 'LineWidth', 4);
hold on
plot(expE.epoch, smoth_testE, 'Color', [1, 0, 0], 'LineWidth', 4);



%% Creating avarage for the graph
handle=[];
j=1;
for i = 1:2:size(exp.epoch)
    handle(i)= exp.epoch(j);
    j=j+1;
end

for i= 2:2:size(exp.epoch)
    handle(i)= exp.train_success_rate(j); 
    j=j+1;
end


[avgH, avgData] = plotAverage(handle, exp.train_success_rate);

% figure(1)
% hold on
% subplot(1,4,3)
% plot(exp_3.epoch, exp_3.train_success_rate, 'Color', [0.9290, 0.6940, 0.1250]);
% title("Exp3")
% xlabel("Epoch")
% ylabel("Train success rate")
% %ylim([0 100])
% set(gca,'FontSize',15)

%% PLOTTING Q - VALUE

figure(6)
plot(exp.epoch, exp.test_mean_Q, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 4);
grid on
title("Mean Q-value - Pick place over the target with demos")
xlabel("Epoch", 'FontSize',35)
ylabel("Mean Q-value training")
set(gca,'FontSize',35)


