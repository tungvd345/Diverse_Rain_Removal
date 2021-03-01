%% This has never been done before
clear;clc;

% Parameter configuration
tic
mode = 'val'; % val
if strcmp(mode,'train')
    startnum = 1;
    endnum = 4000; 
end

if strcmp(mode,'val')
    startnum = 1;
    endnum = 400;
end

datafolder = 'DiverseRainSynthetic';
x = 1;
y = 1;
dx = 512-1;
dy = 512-1;
if ~exist('train', 'file')
    mkdir('train/in/');
    mkdir('train/atm/');
    mkdir('train/trans/');
    mkdir('train/streak/');
    mkdir('train/gt/');
    mkdir('train/light/');
    mkdir('train/medium/');
    mkdir('train/heavy/');
end
if ~exist('val', 'file')
    mkdir('val/in/');
    mkdir('val/atm/');
    mkdir('val/trans/');
    mkdir('val/streak/');
    mkdir('val/gt/');
    mkdir('val/light/');
    mkdir('val/heavy/');
end

if ~exist('filelists', 'file')
    mkdir('filelists/')
end

f_in = fopen(sprintf('filelists/%s_in.txt', mode), 'w');
f_st = fopen(sprintf('filelists/%s_streak.txt', mode), 'w'); 
f_trans = fopen(sprintf('filelists/%s_trans.txt', mode), 'w');
f_atm = fopen(sprintf('filelists/%s_atm.txt', mode), 'w'); 
f_clean = fopen(sprintf('filelists/%s_clean.txt', mode), 'w'); 

%% Render Image

% Set up directory
% root_dir = 'G:\DATASETS\Heavy_rain_image_cvpr2019\CVPR19HeavyRainTrain\test_ver4_monodepthv2';
root_dir = 'G:\New folder\datagen\val';
image_dir = [root_dir, filesep, 'gt', ];
depth_dir = [root_dir , filesep , 'depth'];

image_files = dir([image_dir, filesep, '*.jpg']); 
depth_files = dir([depth_dir, filesep, '*.jpg']); 
num_of_files = length(image_files); 
counter = 1; 

% Render each image
rain_type = 0; % light rain
% for i = 821:820+num_of_files
%     fileindex= i;
%     imname = image_files(i-820).name;
%     depname = strcat('dep_', imname(4:end));
% %     assert(strcmp(imname(1:end-4),depname(1:end-9)));
%     
%     % read image
%     img =im2double(imread([image_files(i-820).folder, filesep, imname]));
%     imwrite((img), sprintf('%s/gt/gt_%04d.jpg', mode, fileindex));
% 
%     depth_img = im2double(imread([depth_files(i-820).folder, filesep, depname])); 
%     imwrite((depth_img), sprintf('%s/depth/dep_%04d.jpg', mode, fileindex));
% 
%     % inverse normalize depth map
%     dep = 1./(depth_img + 1e-6);
%     dep = dep / max(dep(:)); 
%     
% %     imwrite(img(y:y+dy, x:x+dx, :), sprintf('%s/gt/%s', mode, fileindex))
%     
%     for s = 1:5
%         theta = s * 5 + 75;
%         for atmlevel = 4:4
%             tic           
%             %% Render Streak
%             seed = min(1, abs(normrnd(0.5,0.5)));
%             im = imgaussfilt(img, seed);
%             
%             % light rain
%             [rain, streak] = render_rain(im, theta, rain_type, 0.02, 0.8);
%             % medium rain
% %             [rain, streak] = render_rain(im, theta, rain_type, 0.06, 0.7);
%             
%             %% Render Haze
% %             [haze, trans, atm] = render_haze(rain, dep); 
% 
%             %% Crop Image
% %             haze = haze(y:y+dy, x:x+dx, :);
% %             trans = trans(y:y+dy, x:x+dx, :); 
% %             atm = atm(y:y+dy, x:x+dx, :); 
%             rain = rain(y:y+dy, x:x+dx, :); 
%             streak = streak(y:y+dy, x:x+dx, :); 
%             
%             
%             % ======= TO REMOVE ==========
% %             diff = (haze - (1-trans) .* atm )./trans - streak - im(y:y+dy, x:x+dx, :);
% %             if diff > 0.0001
% %                 fprintf('%f, %f, %f', i, s, atmlevel);
% %             end
% 
% %% Write to File
%             
%             imwrite((rain), sprintf('%s/light/light_%04d_s%02d.jpg',mode, fileindex, theta));
% %             imwrite((streak), sprintf('%s/light_streak/light_%04d_s%02d.jpg',mode, fileindex, theta));
% 
% %             imwrite((rain), sprintf('%s/medium/medium_%s_s%02d.jpg',mode, imname(1:end-4), theta));
% %             imwrite((streak), sprintf('%s/medium_streak/medium_%s_s%02d.jpg',mode, imname(1:end-4), theta));
% 
% %             imwrite((haze), sprintf('%s/heavy/heavy_%s.png',mode, imname(1:end-4)));
% 
% %             fprintf(f_in,  sprintf('../../../data/%s/%s/in/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
% %             fprintf(f_trans, sprintf('../../../data/%s/%s/trans/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
% %             fprintf(f_atm, sprintf('../../../data/%s/%s/atm/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
% %             fprintf(f_st, sprintf('../../../data/%s/%s/streak/im_%04d_s%02d_a%02d.png\n', datafolder,mode, fileindex, theta, atmlevel)); 
% %             fprintf(f_clean, sprintf('../../../data/%s/%s/gt/im_%04d.png\n', datafolder, mode, fileindex));
%             
%             fprintf('Num: %d, time elapsed: %f, sigma: %f\n', counter, toc, seed); 
%             counter = counter + 1; 
%         end
%     end
% end

%% medium rain
rain_type = 1;
for i = 821:820+num_of_files
    fileindex= i;
    imname = image_files(i-820).name;
    depname = depth_files(i-820).name;
    
    % read image
    img =im2double(imread([image_files(i-820).folder, filesep, imname]));
%     imwrite((img), sprintf('%s/gt/gt_%04d.png', mode, fileindex));

    depth_img = im2double(imread([depth_files(i-820).folder, filesep, depname])); 
%     imwrite((depth_img), sprintf('%s/depth/dep_%04d.png', mode, fileindex));

    % inverse normalize depth map
    dep = 1./(depth_img + 1e-6);
    dep = dep / max(dep(:)); 
    
%     imwrite(img(y:y+dy, x:x+dx, :), sprintf('%s/gt/%s', mode, fileindex))
    
    for s = 1:5
        theta = s * 5 + 75;
        for atmlevel = 4:4
            tic           
            %% Render Streak
            seed = min(1, abs(normrnd(0.5,0.5)));
            im = imgaussfilt(img, seed);
            
            % light rain
%             [rain, streak] = render_rain(im, theta, dep, 0.02, 0.7);
            % medium rain
            [rain, streak] = render_rain(im, theta, rain_type, 0.06, 0.7);
            
            %% Render Haze
%             [haze, trans, atm] = render_haze(rain, dep); 
            %% Crop Image
%             haze = haze(y:y+dy, x:x+dx, :); 
%             trans = trans(y:y+dy, x:x+dx, :); 
%             atm = atm(y:y+dy, x:x+dx, :); 
            rain = rain(y:y+dy, x:x+dx, :); 
            streak = streak(y:y+dy, x:x+dx, :); 
            
            % ======= TO REMOVE ==========
%             diff = (haze - (1-trans) .* atm )./trans - streak - im(y:y+dy, x:x+dx, :);
%             if diff > 0.0001
%                 fprintf('%f, %f, %f', i, s, atmlevel);
%             end
            
            %% Write to File
            
%             imwrite((rain), sprintf('%s/light/light_%s_s%02d.jpg',mode, imname(1:end-4), theta));
%             imwrite((streak), sprintf('%s/light_streak/light_%s_s%02d.jpg',mode, imname(1:end-4), theta));

            imwrite((rain), sprintf('%s/medium/medium_%04d_s%02d.jpg',mode, fileindex, theta));
%             imwrite((streak), sprintf('%s/medium_streak/medium_%04d_s%02d.jpg',mode, fileindex, theta));

%             imwrite((haze), sprintf('%s/heavy/heavy_%s.png',mode, imname(1:end-4)));
            
%             fprintf(f_in,  sprintf('../../../data/%s/%s/in/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
%             fprintf(f_trans, sprintf('../../../data/%s/%s/trans/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
%             fprintf(f_atm, sprintf('../../../data/%s/%s/atm/im_%04d_s%02d_a%02d.png\n', datafolder, mode, fileindex, theta, atmlevel));
%             fprintf(f_st, sprintf('../../../data/%s/%s/streak/im_%04d_s%02d_a%02d.png\n', datafolder,mode, fileindex, theta, atmlevel)); 
%             fprintf(f_clean, sprintf('../../../data/%s/%s/gt/im_%04d.png\n', datafolder, mode, fileindex));
            
            fprintf('Num: %d, time elapsed: %f, sigma: %f\n', counter, toc, seed); 
            counter = counter + 1; 
        end
    end
end

toc

% 
% gaussdep = imgaussfilt(dep, 100); 
% 
% imshow(gaussdep); 
% beta = 1;
% 
% txmap = exp(-beta * gaussdep); 
% 
% gaussdepdisp = imresize(gaussdep, 0.25); 
% txmapdisp = imresize(txmap, 0.25); 
% 
% imshow([gaussdepdisp; txmapdisp]); 
% 
% fprintf('max: %f, min: %f', max(txmap(:)), min(txmap(:))); 
