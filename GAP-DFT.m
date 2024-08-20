clear all, close all, clc

% Read the data and allocate the arrays
D= readmatrix('data/BE_DFT_APDFT_err.txt','NumHeaderLines',1, 'FileType','text'); 
D= D(:, 3:end);
colors = callmycolors();    %Call color pallet for plotting

elements = ['Cu' 'Pd' 'Ag' 'Pt' 'Au' 'C' 'O'];

mu_P = [1.90 2.20 1.93 2.28 2.54 2.55 3.44]; % Pauling electronegativity [eV]
mu_A = [1.85 1.58 1.87 1.72 1.92 1.262 1.461]; % Allen electronegativity [eV]
mu_aff =[1.235 0.562 1.304 2.125 2.308 2.544 3.610]; % Electron affinity [eV]
Z = [29 46 47 78 79 6 8]; % Atomic number

Z = Z;

y = D(:,3); % all Differences in eV
y_DFT_BE = D(:,1);
y_APDFT_BE = D(:,2);

for i = 1:size(D,1)
    for j=4:4:size(D,2)
        comp(i,j/4)=D(i,j);
        xpos(i,j/4)=D(i,j+1);
        ypos(i,j/4)=D(i,j+2);
        zpos(i,j/4)=D(i,j+3);
    end
end

rcut = 4.0; % Angstrom 
Xb = 4.21931; %box length in the x direction
Yb = 7.3080506326294018;  %box length in the y direction
Rcut = [3.2 9.0];

Natoms = size(xpos,2);
datapoints=size(xpos,1);

C = zeros(datapoints,Natoms,Natoms);

for instance = 1:datapoints                %loop around all the data points (all rows of the x,y,z coordinates)
    for i=1:Natoms                         %loop around all the atoms
        p0 = [xpos(instance,i) ypos(instance,i) zpos(instance,i)];   %save the positions in a list
        for j=i:Natoms                                               %for the each position
            if i < j
                p1 = [xpos(instance,j) ypos(instance,j) zpos(instance,j)];            %get the position
                pp1= [xpos(instance,j)+Xb ypos(instance,j)+Yb zpos(instance,j)]; 
                pp2= [xpos(instance,j)-Xb ypos(instance,j)-Yb zpos(instance,j)];
                pp3= [xpos(instance,j)+Xb ypos(instance,j)-Yb zpos(instance,j)]; 
                pp4= [xpos(instance,j)-Xb ypos(instance,j)+Yb zpos(instance,j)];
                r1 = norm(p1-p0);                                                      %the distance between the atom and all the other atoms
                rpp1 = norm(pp1-p0);
                rpp2 = norm(pp2-p0);
                rpp3 = norm(pp3-p0);
                rpp4 = norm(pp4-p0);
                r = min([r1 rpp1 rpp2 rpp3 rpp4]);
                if comp(instance,i) > 5 || comp(instance,j) > 5
                    rcut = Rcut(2);
                else
                    rcut = Rcut(1);
                end
                if r <= rcut
                    if i==j
                        C(instance,i,j)= 0.5*Z(comp(instance,i))^2.4;
                        % C(instance,i,j)= 0.0;
                    else
                        Zi = Z(comp(instance,i));
                        Zj = Z(comp(instance,j));
                        C(instance,i,j)= (Zi*Zj)/r;
                        C(instance,j,i)= (Zi*Zj)/r;
                    end
                end
            end

        end
    end
end

%% Plot a graph to ensure that everything is alright.
instance = 1;
A(:,:) = C(instance,:,:);
G = graph(A);

myplot = plot(G,'XData',xpos(instance,:),'YData',ypos(instance,:),'ZData',zpos(instance,:),'LineWidth',2)

%% Compute the normalized Laplacian of the graphs and the eigenvalues and vectors.
Lambdas = zeros(datapoints,Natoms);
Vectors = zeros(datapoints,Natoms,Natoms);

% Flatten the data to work with an array.
Data = zeros(datapoints,Natoms*Natoms+Natoms);



for instance = 1:datapoints

    A(:,:) = C(instance,:,:);
    deg = sum(A,1);
    isD = diag(1./sqrt(deg));
    L = eye(Natoms) - isD*A*isD; %Normalized Laplacian
    %L = diag(deg)-A;
    %L = L.^2; %Laplacian
    [V, d] = eig(L,'vector');
    [d, ind] = sort(d);
    V = V(:, ind);

    Lambda(instance,:) = d;
    Vectors(instance,:,:) = V;
    Data(instance,1:Natoms) = d';
end

%% Plot Lambda to understand the situation

plot(Lambda(1,:),'-r','LineWidth',2)
hold on
plot(Lambda(2,:),'-b','LineWidth',2)
plot(Lambda(10,:),'-k','LineWidth',2)
plot(Lambda(20,:),'-c','LineWidth',2)

%% After data was curated, generate the train and validation sets

MAXERR = 0;
M = 50; %number of random cases want to test
%
count=0;
obsv_no = [];
for i=1:M
    Nsamples = size(Data,1); % all cases
    %shuffle = randi([1,Nsamples],Nsamples,1); %shuffle all cases to make random
    shuffle = randperm(Nsamples)';

    pct_train = 80/100;

    Ntrain = round(pct_train*Nsamples); % 80% goes to train
    tsample = shuffle(1:Ntrain); %get the first 80% of cases
    vsample = shuffle(Ntrain+1:end); %get the remaining ones

    % all = (1:Nsamples)';
    % tsample = [1 6 10 16 20 26 30 36 40 46 50 56 60 66 70 76 80]';
    % vsample = all;
    % vsample(tsample) = [];

    xtrain = Data(tsample,:);
    ytrain = y(tsample,1);

    xvalidation = Data(vsample,:);
    yvalidation = y(vsample,1);
    yvalidation_DFT_BE = y_DFT_BE(vsample,1);
    ytrain_DFT_BE = y_DFT_BE(tsample,1);        
    yvalidation_APDFT_BE = y_APDFT_BE(vsample,1);
    ytrain_APDFT_BE = y_APDFT_BE(tsample,1);

    gprMdl1 = fitrgp(xtrain,ytrain);%,'KernelFunction','matern52');
    [ypred1,~,yint1] = predict(gprMdl1,xvalidation);
    [ypred1train,~,yint1train] = predict(gprMdl1,xtrain);

    err = abs(ypred1-yvalidation);
    mymaxerr = max(err);
    if max(err) > MAXERR
        save('GPR_best.mat','gprMdl1');
        MAXERR = mymaxerr;
    end

    maxerr(i,1) = mymaxerr;
    Err = ypred1-yvalidation;
    rmse(i,1) = sqrt(mean(Err).^2);
    mae(i,1) = mean(abs(Err));
    std_err(i,1) = std(Err);
    i;
    count=i+1;
    countr(i,1) = count-1;
    % Append the current value of i to obsv_no
    obsv_no = [obsv_no i];

end


%%
load GPR_best.mat
[ypred1,~,yint1] = predict(gprMdl1,xvalidation);


%%
fig = figure;
fig.Position(3) = fig.Position(3)*2;

tiledlayout(1,1,'TileSpacing','compact')
Nval = size(xvalidation,1);

x = (1:size(xvalidation,1))';
nexttile
hold on
scatter(x,yvalidation,'or','MarkerEdgeColor',[0 .5 .5],'MarkerFaceColor',[0 .7 .7],'LineWidth',1.5)          % Observed data points
scatter(x,ypred1,'sk','MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0 0 0],'LineWidth',1.5)              % GPR predictions
patch([x;flipud(x)],[yint1(:,1);flipud(yint1(:,2))],'k','FaceAlpha',0.05); % Prediction intervals
hold off
title('GPR Fit for binding energy')
legend({'Observations','Predictions','GPR 95%'},'Location','best')
xlabel('Validation Case')
ylabel('Binding energy (eV)')
axis([1 Nval 0 1.0])
set(gca,'FontSize',22)
saveas(gcf,'PredictionConfidence.png')

%%
plotResult(yvalidation, ypred1)


cor_APDFT_BE = yvalidation_APDFT_BE+ ypred1;     %correct APDFT BE for testing 
cor_APDFT_BE_train = ytrain_APDFT_BE + ypred1train;
%%
% scatter(yvalidation*1e3,ypred1*1e3)
% linewidth
lw = 3;

%Marker size
ms=3;

colors=[[0, 0.4470, 0.7410]; [0, 0, 1];
[0.8500, 0.3250, 0.0980];[0, 0.5, 0];
[0.9290, 0.6940, 0.1250];[1, 0, 0];
[0.4940, 0.1840, 0.5560];[0, 0.75, 0.75];
[0.4660, 0.6740, 0.1880];[0.75, 0, 0.75];
[0.3010, 0.7450, 0.9330];[0.75, 0.75, 0];
[0.6350, 0.0780, 0.1840];[0.25, 0.25, 0.25];
[130 78 160]/256; [207 42 40]/256;
[247 144 31]/256; [58 153 69]/256;
[139 69 19]/256;];
m=170;
sz=100;
f4=figure(4);
h=scatter(yvalidation,ypred1, sz, 'filled'); %,err1,'s-','Color',colors(m,:),'LineWidth',lw,'MarkerEdgeColor',colors(m,:),'MarkerFaceColor',colors(m,:),'MarkerSize',ms);


xlabel('Real APDFT error')
ylabel('Predicted APDFT error')
f4.Position = [10 10 1000 1000]; 
set(gca, 'FontSize', 32, 'LineWidth', 2)%, 'xtickLabel', {'Cr','Fe','Co','Ni'})
box on;
%% Visualization of prediction results
% plotResult(yvalidation, yfit)

f8=figure(8);
h1 = scatter(ytrain_DFT_BE, cor_APDFT_BE_train, sz, 'filled', 's'); % 's' is for square markers
hold on
h2 = scatter(yvalidation_DFT_BE, cor_APDFT_BE, sz, 'filled', '^');  % '^' is for triangle markers
 %,err1,'s-','Color',colors(m,:),'LineWidth',lw,'MarkerEdgeColor',colors(m,:),'MarkerFaceColor',colors(m,:),'MarkerSize',ms);
xlabel('BE-DFT (eV)')
ylabel('BE-corrected (eV)')
xticks(-0.5:0.2:1.3)
yticks(-0.5:0.2:1.3)
xlim([-0.5 1.3])
ylim([-0.5 1.3])


f8.Position = [10 10 1000 1000]; 
set(gca, 'FontSize', 32, 'LineWidth', 2)%, 'xtickLabel', {'Cr','Fe','Co','Ni'})
box on;

figure(5)
%plot(rmse(:,1)*1e3,'--sb','LineWidth',1.5,'MarkerEdgeColor',[1 .1 .1],'MarkerFaceColor',[1 .0 .0])
plot(rmse(:,1),'--sb','LineWidth',1.5,'MarkerEdgeColor',[0.8500, 0.3250, 0.0980],'MarkerFaceColor',[1 .0 .0])

%title('Error per case for model 1 and 2')
%legend({'Model 1'},'Location','northeast')
%legend boxoff
xlabel('Random selection')
%ylabel('RMSE [meV]')
ylabel('RMSE (eV)')
set(gca,'FontSize',22)

saveas(gcf,'RMSE.png')
saveas(gcf,'RMSE.eps')

f6=figure(6);
%plot(rmse(:,1)*1e3,'--sb','LineWidth',1.5,'MarkerEdgeColor',[1 .1 .1],'MarkerFaceColor',[1 .0 .0])
plot(mae(:,1),':sb','LineWidth',1.5,'Color',[0, 0.4470, 0.7410], 'MarkerEdgeColor',[0.8500, 0.3250, 0.0980],'MarkerFaceColor',[0.8500, 0.3250, 0.0980])
%lw=2;
%ms=17;
%errorbar(countr(:,1),mae(:,1),std_err(:,1),'--sb','Color',[1 .1 .1],'LineWidth',lw,'MarkerEdgeColor',[1 .0 .0],'MarkerFaceColor',[1 .0 .0],'MarkerSize',ms)
%title('Error per case for model 1 and 2')
%legend({'Model 1'},'Location','northeast')
%legend boxoff
xlabel('Random selection')
%ylabel('RMSE [meV]')
ylabel('MAE (eV)')
set(gca, 'FontSize', 32, 'LineWidth', 2)
xticks(0:10:55);
xlim([0 51]);
yticks(0:0.005:0.04);
ylim([0 0.04]);


f6.Position = [10 10 1000 1000]; 
set(gca, 'FontSize', 28, 'LineWidth', 2)%, 'xtickLabel', {'Cr','Fe','Co','Ni'})
box on;
saveas(gcf,'MAE.png')
saveas(gcf,'MAE.eps')

figure(7)
%plot(maxerr(:,1)*1e3,'--sb','LineWidth',1.5,'MarkerEdgeColor',[1 .1 .1],'MarkerFaceColor',[1 .0 .0])
plot(maxerr(:,1),'--sb','LineWidth',1.5,'MarkerEdgeColor',[1 .1 .1],'MarkerFaceColor',[1 .0 .0])

%stem(obsv_no,mae(:,1)) 
%figure(8)

%title('Max. Error per case for model 1 and 2')
%legend({'Model 1'},'Location','northeast')
%legend boxoff
xlabel('Random selection')
ylabel('Max. Error (eV)')
set(gca,'FontSize',22)

saveas(gcf,'MaxErr.png')
saveas(gcf,'MaxErr.eps')

