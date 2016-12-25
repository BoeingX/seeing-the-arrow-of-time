function [flow] = optical_flow(images, direction)
opticFlow = opticalFlowLK('NoiseThreshold', 0.009);
[optimizer, metric]  = imregconfig('monomodal')
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
    imshow(im2)
end  
end
