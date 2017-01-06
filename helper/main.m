% load data set 1
train1 = textread('./helper/train1.txt', '%s');
test1 = textread('./helper/test1.txt', '%s');

images = dir(strcat('./data/ArrowDataAll/', train1{1}));
images = images(3:end);
opticFlow = opticalFlowLK('NoiseThreshold', 0.009);
%opticFlow = opticalFlowLK();
for i = 3:length(images)
    I = imread(strcat(images(i).folder, '/', images(i).name));
    I = imresize(I, [NaN, 983]);
    IGray = rgb2gray(I);
    flow = estimateFlow(opticFlow, IGray);
    imshow(I)
    hold on
    %plot(flow, 'DecimationFactor', [5 5], 'ScaleFactor', 10)
    %drawnow
    grd      = shiftdim(cat(3,flow.Magnitude,flow.Orientation),2) ;
    grd      = single(grd) ;
    [f, d]        = vl_dsift(grd) ;
    perm = randperm(size(f,2)) ;
    sel = perm(1:50) ;
    h3 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;
    set(h3,'color','g') ;
    hold off
end

%% load data set 2
%train2 = textread('./helper/train2.txt', '%s');
%test2 = textread('./helper/test2.txt', '%s');
%
%% load data set 3
%train3 = textread('./helper/train3.txt', '%s');
%test3 = textread('./helper/test3.txt', '%s');
