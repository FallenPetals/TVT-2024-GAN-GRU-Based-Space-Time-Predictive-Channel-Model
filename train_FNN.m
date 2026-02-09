close all
load 'E:\lza\TWC22\discrete datasets\index_dataset.mat';
% %% LOS
% P1 = LOS_train(:,1:2)'
% T1 = LOS_train(:,3)'
% net = newff(P1,T1,5,{'tansig', 'purelin'}, 'traingd');
% net.trainParam.goal = 1e-5;
% net.trainParam.epochs = 30000;
% net.trainParam.lr = 0.05;
% net.trainParam.showWindow = 1;
% 
% figure;
% X = 1:100
% P2 = LOS_vali(:,1:2)'
% T2 = LOS_vali(:,3)'  %测试结果
% plot(X,T2,'r-+');
% xlabel('Input');
% 
% T3 = net(P2);   %训练结果
% hold on;
% plot(X,T3,'b');
% hold off;
% legend({'Target','Output'})
% 
% 
% rmse = sqrt(mean((T2-T3).^2));
% meap = mean(abs((T2 - T3)./T2))*100



%% NLOS
NP1 = NLOS_train(:,1:2)'
NT1 = NLOS_train(:,3)'
Nnet = newff(NP1,NT1,5,{'tansig', 'purelin'}, 'traingd');
Nnet.trainParam.goal = 1e-5;
Nnet.trainParam.epochs = 30000;
Nnet.trainParam.lr = 0.05;
Nnet.trainParam.showWindow = 1;


figure;
X = 1:100
NP2 = NLOS_vali(:,1:2)'
NT2 = NLOS_vali(:,3)'  %测试结果
plot(X,NT2,'r-+');
xlabel('Input');

NT3 = Nnet(NP2);   %训练结果
hold on;
plot(X,NT3,'b-');
hold off;
legend({'Target','Output'})

Nrmse = sqrt(mean((NT2-NT3).^2));
Nmeap = mean(abs((NT2 - NT3)./NT2))*100
