function BW=segmpp(I, B, athr)
%SEGMPP Segmentation post processing
%   I = original image
%   B = BW mask
%   athr = minimum object size

    % remove small objects
    B=bwareaopen(B, athr);
    L=bwlabel(B);
    %st2=regionprops(B, 'Area', 'PixelIdxList', 'Perimeter', 'Solidity');
    st2=rprops(B, 'Area', 'PixelIdxList', 'Solidity');
    BW=false(size(B));
    BW2=BW;
    
    % threshold each region
    for i=1:numel(st2)
        it1=st2(i);
        pxl=it1.PixelIdxList;
        BW(pxl)=I(pxl)>=multithresh(I(pxl));
    end
    
    %st3=regionprops(BW, 'Area', 'PixelIdxList', 'Perimeter', 'Solidity');
    st3=rprops(BW, 'Area', 'PixelIdxList', 'Solidity');
    mods=zeros(1, max(L(:)));
    
    for i=1:numel(st3)
        it=st3(i);
        % find region which this object belongs to
        [Ic, by, bx, M]=ind2rect(L, it.PixelIdxList);
        uu=sort(unique(Ic));
        ll=uu(end);
        xx1=bx(1):bx(2);
        yy1=by(1):by(2);
        
        % replace if parents convexity is smaller
        if (st2(ll).Solidity < it.Solidity) && it.Area > athr
            BW2(yy1, xx1)=BW2(yy1, xx1) | M;
            mods(ll)=mods(ll)+1;
        end
    end
    
    % put back orig subimage if there were no modifications
    for i=1:numel(st2)
        if mods(i) == 0
            pxl=st2(i).PixelIdxList;
            BW2(pxl)=true;
        end
    end
    BW=BW2;
end