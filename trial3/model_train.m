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
% adam optimizer
beta1=0.95;
beta2=0.999;
eps_adam=1e-8;
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
        mW{l}=gpuArray(single(mW{l}));
        mb{l}=gpuArray(single(mb{l}));
        vW{l}=gpuArray(single(vW{l}));
        vb{l}=gpuArray(single(vb{l}));
    end
else
    % initialization
    epoch0=0;
    W=cell(L,1);
    b=cell(L,1);
    mW=cell(L,1);
    mb=cell(L,1);
    vW=cell(L,1);
    vb=cell(L,1);
    for l=2:L
        W{l}=gpuArray(single(randn(n(l),n(l-1))*sqrt(2/n(l-1)))); % He init
        b{l}=gpuArray(single(randn(n(l),1)));
        mW{l}=gpuArray(single(zeros(n(l),n(l-1))));
        mb{l}=gpuArray(single(zeros(n(l),1)));
        vW{l}=gpuArray(single(zeros(n(l),n(l-1))));
        vb{l}=gpuArray(single(zeros(n(l),1)));
    end
    t_adam=0;
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
            a{l}=acti_sigmoid(z{l});
        end
        z{L}=W{L}*a{L-1}+b{L};
        a{L}=acti_softmax(z{L}); % output layer
        
        % backward propagation
        d=cell(L,1);
        d{L}=(a{L}-yi_batch)/batchSize; % output layer (cross-entropy + softmax)
        for l=L-1:-1:2 % hidden layers (sigmoid derivative)
            d{l}=(W{l+1}' * d{l+1}).*(a{l}.*(1-a{l}));
        end

        % compute gradient
        dW=cell(L,1);
        db=cell(L,1);
        for l=2:L
            dW{l}=d{l}*a{l-1}';
            db{l}=sum(d{l},2);
        end
        
        % adam optimizer
        t_adam=t_adam+1;
        for l=2:L
            % update mean
            mW{l}=beta1*mW{l}+(1-beta1)*dW{l};
            mb{l}=beta1*mb{l}+(1-beta1)*db{l};
            % update variance
            vW{l}=beta2*vW{l}+(1-beta2)*dW{l}.^2;
            vb{l}=beta2*vb{l}+(1-beta2)*db{l}.^2;
            % bias correction
            mW_hat=mW{l}/(1-beta1^t_adam);
            mb_hat=mb{l}/(1-beta1^t_adam);
            vW_hat=vW{l}/(1-beta2^t_adam);
            vb_hat=vb{l}/(1-beta2^t_adam);
            % update parameters
            W{l}=W{l}-lr*mW_hat./(sqrt(vW_hat)+eps_adam);
            b{l}=b{l}-lr*mb_hat./(sqrt(vb_hat)+eps_adam);
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
    mW_tmp = mW; 
    mb_tmp = mb;
    vW_tmp = vW; 
    vb_tmp = vb;
    for l = 2:L
        W{l} = gather(W{l});
        b{l} = gather(b{l});
        mW{l} = gather(mW{l});
        mb{l} = gather(mb{l});
        vW{l} = gather(vW{l});
        vb{l} = gather(vb{l});
    end
    filename = sprintf('%s/WB%04d',folder_name, epoch);
    save(filename, "W", "b", "epoch", "cost_list", "mW", "vW", "mb", "vb", "t_adam", "lr", "best_cost", "lr_wait");
    W = W_tmp;
    b = b_tmp;
    mW = mW_tmp;
    vW = vW_tmp;
    mb = mb_tmp;
    vb = vb_tmp;
end

training_time = toc;
fprintf("Training time: %.2f s\n", training_time);