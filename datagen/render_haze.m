function [haze, trans, A] = render_haze(img, dep)

h = size(img, 1);
w = size(img, 2);

% transmittance
gaussdep = imgaussfilt(dep, 5); 
% gaussdep = imguidedfilter(dep);
% for train
% beta = rand() * 0.8 + 0.8;
beta = 2;
% for val
% beta = rand() * 2 + 2.6;

tx = exp(-beta * gaussdep); 
%fprintf('%f %f %f %f', beta, max(tx(:)), min(tx(:)), mean(tx(:)));

% atmospheric light
% for train
a = 0.8;%*rand();
% a = 0;
% for val
% a = 0.25 + 0.7 * rand();
A = ones(h,w, 3) * a; 
trans = repmat(tx, [1,1,3]); 
% trans = ones(h, w, 3);
% Render
haze = img .* trans + (1-trans) .* A;

end

