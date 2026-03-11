%% convert given result (NUMBER) into output layer
%  num : 0 -> [1 0 0 0 0 0 0 0 0 0]';
%  num : 1 -> [0 1 0 0 0 0 0 0 0 0]';
%  num : 2 -> [0 0 1 0 0 0 0 0 0 0]';
%
%  num : 8 -> [0 0 0 0 0 0 0 0 1 0]';
%  num : 9 -> [0 0 0 0 0 0 0 0 0 1]';
function res = num2arr(n)
    res=zeros(10,1);
    for k=1:10
        if n==k-1
            res(k)=1;
            break;
        end
    end
end
