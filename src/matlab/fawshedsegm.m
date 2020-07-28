function [BW] = fawshedsegm(I, sthr, thrmethod)
% I = Intensity image to be segmented
% sthr= minimum size of object
% thrmethod = Thresholding method. Either 'AdaptiveWatershed', 'Adaptive'
% or 'Basic'

% Abdolhoseini, M., Kluge, M.G., Walker, F.R. and Johnson, S.J., 2019. Segmentation of heavily clustered nuclei from histopathological images. Scientific reports, 9(1), pp.1-13.

if nargin < 2
    sthr=75;
end

if nargin < 3
    thrmethod='wthresh';
end

if strcmp(thrmethod, 'adaptive')
    T=adaptthresh(I);
    BW1=imbinarize(I, T);
elseif strcmp(thrmethod, 'wthresh')
    BW1=wthresh(I, sthr);
elseif strcmp(thrmethod, 'basic')
    BW1=imbinarize(I);
else
    error('Invalid method');
end

% split clustered nuclei
BW2=splitnuclei(I, BW1, sthr);
BW3=mergenuclei(BW2, sthr);
%BW3=mergenuclei(bwfill(BW2, 'holes'), sthr);
% post process
%BW3=bwfill(BW3, 'holes');
BW4=bwfill(segmpp(I, BW3, sthr), 'holes');
BW=BW4;

end
