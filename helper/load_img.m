function [I] = load_img(image, isflip)
I = imread(strcat(image.folder, '/', image.name));
%I = double(imresize(I, [NaN, 983]));
I = imresize(I, [NaN, 983]);
if isflip
    I = flip(I, 2);
end
end
