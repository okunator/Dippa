% [C, by, bx, M]= ind2rect(I, ix)
% Extract subimage from image given indices

% I = image
% ix = indices inside subregion
% outs:
% C = extracted image
% by = bounding box Y-range
% bx = bounding box X-range
% M = mask
function [C, by, bx, M]= ind2rect(I, ix)
    xy=size(I);
    xy=xy(1:2);

    [yy, xx]= ind2sub(xy, ix);
    by= [min(yy); max(yy)];
    bx= [min(xx); max(xx)];    
    
    yy2=yy-min(yy)+1;
    xx2=xx-min(xx)+1;
     
    M=false(max(yy2), max(xx2));
    M(sub2ind(size(M), yy2, xx2))=1;   
    try
        if numel(size(I)) == 2
            C=I(by(1):by(2), bx(1):bx(2));
            C(~M)=0;
        else
            C=I(by(1):by(2), bx(1):bx(2), :);
            C=imask(C, ~M);
        end
    catch
       'error' 
    end
    %C(~M)=0;
end