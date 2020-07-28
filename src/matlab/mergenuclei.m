function [M] = mergenuclei(BW, wmin)
%mergenuclei Attempt to fix oversegmented nuclei from H&E image
%   BW = ROI mask
%   wmin = minimum object size
% 
%   Result
%   M = mask

thr=wmin*10;
L1=bwlabel(BW);
% get all objects
%R1=regionprops(L1, 'Area', 'Perimeter', 'PixelIdxList', 'Solidity');
R1=rprops(BW, 'Area', 'PixelIdxList', 'Solidity');

NN=bwneigh(BW);
M=BW;

done=false(numel(NN), 1);
% 1st pass: check groups with more than 2 objs
for i=1:numel(NN)
    gg=intersect([i; get_group(NN, i)], find(~done));
    
    if numel(gg) > 2
        ix={R1(gg).PixelIdxList};
        [M1,X]=merge_objs(size(M), cat(1, ix{:}));
        
        if numel(X) > 1
           warning('Should have one object but found %d', numel(X))
           continue
        end
        if X.Area < thr && all(X.Solidity > [R1(gg).Solidity])
           M(M1)=true;
           done(gg)=true;
        else
           
        end
    end
end

% 2nd pass: check pairs
sel=find(~done);
for i=1:numel(sel)
    ss=sel(i);
    
    gg=sort(intersect([NN{ss}], find(~done)));
    if done(ss) || isempty(gg), continue, end
    assert(any(gg==ss))
    
    % iterate through neighbours
    for j=2:numel(gg)
        ix2=[gg(1) gg(j)];
        ix={R1(ix2).PixelIdxList};
        [M1,X]=merge_objs(size(M), cat(1, ix{:}));
        
        if numel(X) > 1
           warning('Should have one object but found %d', numel(X))
           continue
        end
        
        if X.Area < thr && (X.Solidity > mean([R1(ix2).Solidity]))
           M(M1)=true;
           done(ix2)=true;
        end
    end
    
    done(ss)=true;
end

end

function [M,R]=merge_objs(sz, pxl)
    M=false(sz);
    [y, x]=ind2sub(sz, pxl);
    ll=min([y, x]);
    lu=max([y, x]);
    xx=ll(2):lu(2);
    yy=ll(1):lu(1);
    % get subimage
    M0=false(numel(yy), numel(xx));
    K=[y, x]-[(ll(1)-1) (ll(2)-1)];
    ix=sub2ind(size(M0), K(:, 1), K(:, 2));
    M0(ix)=pxl;
    M0=bwmorph(M0, 'bridge');
    %R=regionprops(M0, 'Area', 'Solidity');
    R=rprops(M0, 'Area', 'Solidity');
    M(yy, xx)=M0;
end

function inds=get_group(NN, ix)
    checked_nodes=false(numel(NN), 1);
    q=NN{ix};
    while numel(q) > 0
        % get neighbour nodes not checked already
        n=intersect(q, find(~checked_nodes));
        if isempty(n), break, end
        
        % mark as checked
        checked_nodes(n)=true;
        % get next nodes
        q=NN{n};
    end
    inds=find(checked_nodes);
end
