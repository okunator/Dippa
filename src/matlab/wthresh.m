function B= wthresh(I, smin)
% Adaptive watershed based thresholding
% I = input image (grayscale)
% smin = minimum object size

    if nargin < 2
        smin=5;
    end

    % Threshold globally
    B1=I>=multithresh(I);

    % Apply watershed, to find ROI:s
    GL=watershed(~B1); 
    B=false(size(I));
    
    st=regionprops(GL, 'Area', 'PixelIdxList', 'BoundingBox');
    
    % rebuild mask by thresholding each region
    for i= 1:numel(st)
       ix1=st(i).PixelIdxList;
       if st(i).Area > smin*0.1
           [G2b, ~, ~, M]=ind2rect(I, ix1);
           G2b=G2b>=multithresh(I(ix1));
           B(ix1)=G2b(M);
           %B(yb(1):yb(2), xb(1):xb(2))= B(yb(1):yb(2), xb(1):xb(2)) | G2b; 
       end
    end
end
