% I = imread('toy_src/im_0601.png');
% % J = imnoise(I,'gaussian',0.005, 0.01);
% J = imnoise(I, 'salt & pepper', 0.1);
% imshow(J)
% imwrite(J,'trial.png');

% root_dir = 'G:\DATASETS\DID-MDN-datasets\DID-MDN-training\Rain_Light\train2018new';
root_dir = 'G:\DATASETS\DID-MDN-datasets\DID-MDN-test';
image_dir = [root_dir, filesep, 'toy_src', ];
depth_dir = [root_dir , filesep , 'toy_depth'];

image_files = dir([root_dir, filesep, '*.jpg']);
 
for i = 1:4000
    img_name = image_files(i).name;
    img_in_gt = imread([image_files(i).folder, filesep, img_name]);
    img_gt = img_in_gt(:,513:1024,:);
    imwrite(img_gt, sprintf('DID-MDN_orig/DID-MDN_test_orig/%s',img_name));
    fprintf('image %s: \n',img_name);
end