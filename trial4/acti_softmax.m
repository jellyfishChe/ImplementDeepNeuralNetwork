function y=acti_softmax(x)
    x=x-max(x,[],1);
    y=exp(x)./(sum(exp(x),1)+1e-10);
end