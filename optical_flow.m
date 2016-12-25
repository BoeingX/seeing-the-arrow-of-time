function [flows] = optical_flow(images, direction)
flows = [];
[optimizer, metric]  = imregconfig('monomodal');
if strcmp(direction, 'A')
    isflip = false;
elseif strcmp(direction, 'B')
    isflip = true;
elseif strcmp(direction, 'C')
    images = flip(images);
    isflip = false;
else
    images = flip(images);
    isflip = true;
end
for i = 2:(length(images)-1)
    im1 = load_img(images(i), isflip);
    im2 = load_img(images(i), isflip);
    im3 = load_img(images(i), isflip);
    im1_gray = rgb2gray(im1);
    im2_gray = rgb2gray(im2);
    im3_gray = rgb2gray(im3);
    tform12 = imregtform(im1_gray, im2_gray, 'rigid', optimizer, metric);
    tform23 = imregtform(im2_gray, im3_gray, 'rigid', optimizer, metric);
    im12 = imwarp(im1,tform12);
    im23 = imwarp(im3,tform23);
    flow = mex_OF(im12, im23);
    flows = [flows; flow];
end  
end
