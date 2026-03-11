clc; clear; close all;
rng(2,"twister");
load mnist.mat;
img_train=training.images;
imgLen=training.count;
%% setup parameters
n=[784 64 32 10];
L=length(n);
lr=1e-3;
max_epoch=100;
cost_list=[];
% batch size
batchSize=64;
% reduce learning rate on plateau
lr_patience=5;
lr_factor=0.5;
lr_min=1e-7;
%% GPU setting
gpu=gpuDevice;
fprintf("Using GPU: %s\n",gpu.Name);

%% data pre-processing
fprintf("Data Pre-processing\n");
xi=reshape(img_train,[784,imgLen]);
yi=zeros(10,imgLen);
for i=1:imgLen
    yi(:,i)=num2arr(training.labels(i));
end
xi=gpuArray(single(xi));
yi=gpuArray(single(yi));
%% load history parameters
folder_name="WBtrain";
if ~exist(folder_name,"dir")
    mkdir(folder_name);
end
% hist=dir("./WBtrain/*.mat");
hist=dir(sprintf("./%s/*.mat",folder_name));
if ~isempty(hist)
    load(sprintf("%s/WB%04d.mat",folder_name,length(hist)));
    epoch0=epoch;
    for l=2:L
        W{l}=gpuArray(single(W{l}));
        b{l}=gpuArray(single(b{l}));
    end
else
    % initialization
    epoch0=0;
    W=cell(L,1);
    b=cell(L,1);
    for l=2:L
        W{l}=gpuArray(single(randn(n(l),n(l-1))*sqrt(2/n(l-1)))); % He init
        b{l}=gpuArray(single(randn(n(l),1)));
    end
    best_cost=Inf;
    lr_wait=0;
end

%% training
batches=floor(imgLen/batchSize);
tic;
for epoch=1+epoch0:max_epoch+epoch0
    perm=randperm(imgLen);
    xi_p=xi(:,perm);
    yi_p=yi(:,perm);
    for batch_i=1:batches
        b_left=(batch_i-1)*batchSize+1;
        b_right=batch_i*batchSize;
        xi_batch=xi_p(:,b_left:b_right);
        yi_batch=yi_p(:,b_left:b_right);
        
        % forward pass
        a=cell(L,1);
        z=cell(L,1);
        a{1}=xi_batch; % input layer
        for l=2:L-1 % hidden layers
            z{l}=W{l}*a{l-1}+b{l};
            a{l}=acti_relu(z{l});
        end
        z{L}=W{L}*a{L-1}+b{L};
        a{L}=acti_softmax(z{L}); % output layer
        
        % backward propagation
        d=cell(L,1);
        d{L}=(a{L}-yi_batch)/batchSize; % output layer
        for l=L-1:-1:2 % hidden layers
            d{l}=(W{l+1}' * d{l+1}).*(z{l}>0);
        end

        % compute gradient
        dW=cell(L,1);
        db=cell(L,1);
        for l=2:L
            dW{l}=d{l}*a{l-1}';
            db{l}=sum(d{l},2);
        end
        
        % SGD optimizer
        for l=2:L
            W{l}=W{l}-lr*dW{l};
            b{l}=b{l}-lr*db{l};
        end
    end
    clc;
    % compute cost
    costVal=compute_cost(W,b,xi,yi,L);
    cost_list = [cost_list, costVal];

    % reduce learning rate on plateau
    if costVal<best_cost
        best_cost=costVal;
        lr_wait=0;
    else
        lr_wait=lr_wait+1;
        if lr_wait>=lr_patience && lr>lr_min
            lr=max(lr*lr_factor,lr_min);
            lr_wait=0;
            fprintf("lr reduced\n");
        end
    end
    
    fprintf("Epoch %d / %d\n",epoch,max_epoch+epoch0);
    fprintf(" Cost = %.2e | lr = %.2e\n",costVal,lr);
    % saving checkpoint
    W_tmp = W; 
    b_tmp = b; 
    for l = 2:L
        W{l} = gather(W{l});
        b{l} = gather(b{l});
    end
    filename = sprintf('%s/WB%04d',folder_name, epoch);
    save(filename, "W", "b", "epoch", "cost_list", "lr", "best_cost", "lr_wait");
    W = W_tmp;
    b = b_tmp;
end

training_time = toc;
fprintf("Training time: %.2f s\n", training_time);