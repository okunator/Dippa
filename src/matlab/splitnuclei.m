function [I] = splitnuclei(O, B, thr)
%SPLITNUCLEI Summary of this function goes here
%   O = original image
%   B = adaptively thresholded BW image (i.e. via wshedthresh)
%   thr = minimum object size
%   

st2=regionprops(B, 'Area', 'BoundingBox', 'PixelIdxList');
st3=st2([st2.Area] > thr );
I=false(size(B));

for i= 1:numel(st3)
    % Get cropped image
    [Ic, by, bx, Gc]=ind2rect(O, st3(i).PixelIdxList);
    yl=by(1):by(2);
    xl=bx(1):bx(2);
    Ic=imcomplement(Ic);
    D=bwdist(~Gc);
    
    % Normalize values in distance map and cropped image
    Dn=dmap_scale(D, Gc);
    In=dmap_scale(Ic, Gc);
    S=In - Dn;
    assert(~any(isnan(S(:))));
    nthr=min(ceil(st3(i).Area/thr), 20);
    thr1=multithresh(S(Gc), nthr);
    assert(~any(isnan(thr1(:))));
    IM=ones(size(S))*(nthr+2);
    Iq=imquantize(S, thr1);
    
    IM(Gc)=Iq(Gc);
    
    % clean map
    for j= 1:nthr
        L=IM == j;
        % threshold for moving to next level
        thr2=j*thr/(nthr+1);
        % put small objects to next level
        Lh=extract_small_pieces(L, thr2);
        IM(Lh)=j+1;
    end
    
    W=watershed(IM);
    Gc(W==0)= 0;
    
    % store back to original image
    I(yl, xl)=Gc | I(yl, xl);
end
end

function I=dmap_scale(I, M)
    I=single(I);
    pxl=I(M);
    mi=min(pxl);
    ma=max(pxl);
    if mi ~= ma
        I=(I - mi)/(ma -mi);
    else
        I=I/mi;
    end
end

function I= extract_small_pieces(L, p)
    oo= bwconncomp(L);
    I= false(numel(L), 1);
    for i= 1:oo.NumObjects
       if numel(oo.PixelIdxList{i}) < p
           I(oo.PixelIdxList{i})= true;
       end
    end
    I= reshape(I, size(L));
end
