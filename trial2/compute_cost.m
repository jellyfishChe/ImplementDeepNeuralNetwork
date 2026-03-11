% compute the cost value by cross entropy loss
function cost=compute_cost(W,b,xi,yi,L)
    a=xi;
    for l=2:L-1
        a= acti_relu(W{l}*a+b{l});
    end
    a=acti_softmax(W{L}*a+b{L});
    N=size(xi,2);
    cost=-1/N*sum(sum(yi.*log( a+1e-8 )));
end