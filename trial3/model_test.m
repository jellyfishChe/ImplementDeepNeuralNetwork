clc;clear;close all;
n=[784 64 32 10];
L=length(n);
load mnist.mat;
folder_name="WBtrain";
hist=dir(sprintf("./%s/*.mat",folder_name));
if isempty(hist)
    error("no model to test");
end
selectedModel=length(hist);
load(sprintf("%s/WB%04d.mat",folder_name,selectedModel));

img_test=reshape(test.images,[784,test.count]);
num_test=test.labels;
imgLen=test.count;

%% forward pass
a=img_test;
for l=2:L-1
    a=acti_sigmoid(W{l}*a+b{l});
end
a=acti_softmax(W{L}*a+b{L});

%% compute accuracy rate
acc=0;
for iter=1:imgLen
    if arr2num(a(:,iter))==num_test(iter)
        acc=acc+1;
    end
end
fprintf("Accuracy: %.4f (%d/%d)\n",acc/imgLen,acc,imgLen);

%% plot cost curve
figure;
semilogy(cost_list);
%plot(cost_list);
grid on;
xlabel("Epoch");
ylabel("Cost Value");
title("Training Loss");