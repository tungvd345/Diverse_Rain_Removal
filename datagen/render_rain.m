function [image_rain, actual_streak] = render_rain(img,  theta, rain_type, density, intensity)

image_rain = img;
h = size(img, 1); 
w = size(img, 2); 

% parameter seed gen

% s = 1.01+rand() * 0.2;
% m = (density) *(0.2+ rand()*(0.05)); % mean of gaussian, controls density of rain
% v = intensity + rand(1,1)*0.3; % variance of gaussian,  controls intensity of rain streak
% l = randi(40) + 20; % len of motion blur, control size of rain streak
s_far = 1.2 + rand()*0.3;
s_near = 3.4 + rand()*0.2;
m_far = density + rand()*0.02; % mean of gaussian, controls density of rain
m_near = density/10 + rand()*0.002;
v = 0.2; % variance of gaussian,  controls intensity of rain streak
l = randi(60) + 20;

% Generate proper noise seed

dense_chnl = zeros(h,w, 1);
% dense_chnl_noise = imnoise(dense_chnl, 'gaussian', m, v);
dense_chnl_noise_far = imnoise(dense_chnl, 'salt & pepper', m_far);
dense_chnl_noise_far = imresize(dense_chnl_noise_far, s_far, 'bicubic'); 
posv_far = randi(size(dense_chnl_noise_far, 1) - h); 
posh_far = randi(size(dense_chnl_noise_far, 2) - w);
dense_chnl_noise_far = dense_chnl_noise_far(posv_far:posv_far+h-1, posh_far:posh_far+w-1);
% dense_chnl_noise_far(dep<0.1) = 0;
% form filter
filter = fspecial('motion', l, theta);
dense_chnl_motion_far = imfilter(dense_chnl_noise_far, filter);

dense_chnl_noise_near = imnoise(dense_chnl, 'salt & pepper', m_near);
dense_chnl_noise_near = imresize(dense_chnl_noise_near, s_near, 'bicubic'); 
posv_near = randi(size(dense_chnl_noise_near, 1) - h); 
posh_near = randi(size(dense_chnl_noise_near, 2) - w);
dense_chnl_noise_near = dense_chnl_noise_near(posv_near:posv_near+h-1, posh_near:posh_near+w-1);
% form filter
filter = fspecial('motion', l, theta);
dense_chnl_motion_near = imfilter(dense_chnl_noise_near, filter);

if rain_type == 0
    dense_chnl_motion = dense_chnl_motion_far;
else
    dense_chnl_motion = dense_chnl_motion_near + dense_chnl_motion_far;
end
% Generate streak with motion blur
dense_chnl_motion(dense_chnl_motion<0) = 0;
dense_streak = repmat(dense_chnl_motion, [1,1,3]); 

% Render Rain streak
% tr = 0.25 + rand()*0.05;
tr = 1.2 + rand()*0.3;
image_rain = image_rain + tr*dense_streak ;%+ 0.05 - 0.3*rand(); % render dense rain image
image_rain(image_rain >= 1) = 1;
image_rain(image_rain < 0) = 0;
actual_streak = abs(image_rain - img); 
end
