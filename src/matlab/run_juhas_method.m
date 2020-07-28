
H=imread('/home/leos/IMAGE_ANALYSIS/segm/prob_map.png');
BW=fawshedsegm(H, 200, 'wthresh');

%%
se = strel('disk',5);
pat='/home/leos/IMAGE_ANALYSIS/model_ouputs/Kumar/probability_maps/For_matlab/unet_smp/';  % Your folder
fil=fullfile(pat,'*.png');
d=dir(fil);
for k=1:numel(d)
  filename=fullfile(pat,d(k).name)
  % do
  H=imread(filename);
  BW=fawshedsegm(H, 200, 'wthresh');
  
  new_name = strcat(d(k).folder, strrep(d(k).name,'prob_map.png','mask.png'));
  imwrite(BW, new_name);
end
%%
d(1).name
d(1).folder