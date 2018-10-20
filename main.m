clear
clc
dataset=readtable('Gasification_Dataset_y2.csv');
[X,y] = Preprocessing(dataset);

totaldata(:,1:13)=X;
totaldata(:,14)=y;

totaldata = totaldata(randperm(size(totaldata,1)),:);

indices = crossvalind('Kfold', 56, 10);
for i=1:10
    test = (indices == 1); training = ~test;
    [weight1,weight2]=MLP(totaldata(training,1:13),totaldata(training,14)); %Train 
    
    X=totaldata(test,1:13);
    X(:,14)=1;
    a2 = g(X * weight1');
    a2(:,6)=1;
    a3 = 0.01 * (a2 * weight2');
    
    y=totaldata(test,14);
    
    sse=sum( (a3-y).^2 );
    sst=sum( (y - mean(y)).^2 );
    r2=1-(sse/sst);
    rmse=sqrt( mean((a3-y).^2) );

    V(i)=rmse;
end

%%Trial boi

clearvars -except V
function sigmoid = g(x)

    sigmoid = 1./(1+exp(-x));
end

%%Trial boi
