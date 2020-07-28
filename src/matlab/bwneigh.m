function [N] = bwneigh(BW)
%%BWNEIGH Find neighboring objects from BW image

[L, b]=bwlabel(BW);
b=b+1;
% Use convolution just for fun.
% Left/right/top/bottom
C=conv2(L, [0 1 0; b 0 b^2; 0 b^3 0], 'valid');
nn=unique(C(:));
bb=dec2abase(nn, b);
bb=unique(bb, 'rows');
N=cell(1, b);
for i=1:b
    N{i}=unique(bb(any(bb==i, 2), :));
end

end

function R=dec2abase(M, b)
%% convert M from base 10 to base b
% M = x_n * b^n + x_(n-1) + b^(n-1) + ... + x_1*b^1 + x_0*b^0
%
x=max(ceil(loga(M, b)));
R=zeros(numel(M), x);

for i=1:size(R, 2)
    e=x-i;
    i1=floor(M/b^e);
    R(:, i)=i1;
    M=M-i1*b^e;
end

end

function x=loga(x, b)
x=log2(x)/log2(b);
end