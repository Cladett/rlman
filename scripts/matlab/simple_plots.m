%% PLOTTING RESULTS FROM TRAINING OF AGENT progress.csv %%


experiment_1 = 'dVRL_HER_4';
exp_1 = readtable(strcat(experiment_1, '/', 'progress.csv'));

experiment_2 = 'dVRL_HER_3';
exp_2 = readtable(strcat(experiment_2, '/', 'progress.csv'));

experiment_3 = 'dVRL_HER_2';
exp_3 = readtable(strcat(experiment_3, '/', 'progress.csv'));

experiment_4 = 'dVRL_HER';
exp_4 = readtable(strcat(experiment_4, '/', 'progress.csv'));

x0 = 10;
y0 = 10;
width = 1000;
height = 500;

% Compare evaluation reward, training error
figure(1);
subplot(1,3,1)
%top = (exp_1.rollout_errors + exp_1.rollout_errors_dev) * 1000;
%bottom = (exp_1.rollout_errors - exp_1.rollout_errors_dev) * 1000;
%shade(exp_1.total_steps, top, exp_1.total_steps, bottom, 'Color', [0.8500, 0.3250, 0.0980],'FillType', [1 2])
hold on
plot(exp_2.epoch, exp_2.train_success_rate, 'Color', [0.8500, 0.3250, 0.0980]);
title("Constant")
xlabel("Epoch")
ylabel("Train success rate")
ylim([0 100])
set(gca,'FontSize',15)

subplot(1,3,2)
top = (exp_2.rollout_errors + exp_2.rollout_errors_dev) * 1000;
bottom = (exp_2.rollout_errors - exp_2.rollout_errors_dev) * 1000;
shade(exp_2.total_steps, top, exp_2.total_steps, bottom, 'Color', [0, 0.4470, 0.7410],'FillType', [1 2])
hold on
plot(exp_2.total_steps, exp_2.rollout_errors * 1000, 'Color', [0, 0.4470, 0.7410]);
title("Linear")
xlabel("Training steps")
ylabel("Error (mm)")
ylim([0 100])
set(gca,'FontSize',15)

subplot(1,3,3)
top = (exp_3.rollout_errors + exp_3.rollout_errors_dev) * 1000;
bottom = (exp_3.rollout_errors - exp_3.rollout_errors_dev) * 1000;
shade(exp_3.total_steps, top, exp_3.total_steps, bottom, 'Color', [0.9290, 0.6940, 0.1250], 'FillType', [1 2])
hold on
plot(exp_3.total_steps, exp_3.rollout_errors * 1000, 'Color', [0.9290, 0.6940, 0.1250]);
title("Decay")
xlabel("Training steps")
ylabel("Error (mm)")
ylim([0 100])
set(gca,'FontSize',15)
set(gcf,'position',[x0,y0,width,height])

figure;
plot(exp_1.total_steps, exp_1.rollout_return, 'Color', [0.8500, 0.3250, 0.0980], 'LineWidth', 2.5)
hold on
plot(exp_2.total_steps, exp_2.rollout_return, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 2.5)
hold on
plot(exp_3.total_steps, exp_3.rollout_return, 'Color', [0.9290, 0.6940, 0.1250], 'LineWidth', 2.5)
set(gca,'FontSize',18)
xlabel("Training steps")
ylabel("Episode Reward")
legend('Constant', 'Decay')



