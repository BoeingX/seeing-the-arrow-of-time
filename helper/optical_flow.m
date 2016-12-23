function [flow] = optical_flow(im1, im2, im3)
tmp = mex_OF(double(im1), double(im2));
flow.Vx = tmp(:, :, 1);
flow.Vy = tmp(:, :, 2);
flow.Magnitude = sqrt((flow.Vx).^2 + (flow.Vy).^2);
flow.Orientation = atan2(flow.Vy, flow.Vx);
end
