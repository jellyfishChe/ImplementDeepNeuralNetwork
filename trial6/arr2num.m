%% convert output layer into the NUMBER 0 ~ 9
function n=arr2num(A) % A must be 10*1
    for k=1:10
        if A(k)==max(A)
            n=k-1;
        end
    end
end