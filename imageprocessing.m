%%第一版

function ImageProcessingGUI
    % 创建图形界面
    hFig = figure('Name', '图像处理系统', 'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', ...
                  'Position', [100, 100, 800, 600]);

    % 添加控件
    uicontrol('Style', 'pushbutton', 'String', '打开图像', ...
              'Position', [50, 550, 100, 30], ...
              'Callback', @openImageCallback);

    uicontrol('Style', 'pushbutton', 'String', '灰度直方图 & 均衡化', ...
              'Position', [50, 500, 150, 30], ...
              'Callback', @histogramCallback);

    uicontrol('Style', 'pushbutton', 'String', '对比度增强', ...
              'Position', [50, 450, 100, 30], ...
              'Callback', @contrastEnhancementCallback);

    uicontrol('Style', 'pushbutton', 'String', '几何变换', ...
              'Position', [50, 400, 100, 30], ...
              'Callback', @geometricTransformCallback);

    uicontrol('Style', 'pushbutton', 'String', '图像去噪', ...
              'Position', [50, 350, 100, 30], ...
              'Callback', @denoiseCallback);

    uicontrol('Style', 'pushbutton', 'String', '边缘检测', ...
              'Position', [50, 300, 100, 30], ...
              'Callback', @edgeDetectionCallback);

    uicontrol('Style', 'pushbutton', 'String', '目标提取', ...
              'Position', [50, 250, 100, 30], ...
              'Callback', @objectExtractionCallback);

    uicontrol('Style', 'pushbutton', 'String', '特征提取', ...
              'Position', [50, 200, 100, 30], ...
              'Callback', @featureExtractionCallback);

    % 添加显示图像的轴
    axes('Units', 'pixels', 'Position', [250, 150, 500, 400], 'Tag', 'ImageAxes');
end

% 打开图像
function openImageCallback(~, ~)
    [file, path] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件(*.jpg, *.png, *.bmp)'});
    if isequal(file, 0)
        return;
    end
    img = imread(fullfile(path, file));
    assignin('base', 'img', img); % 保存到工作区
    displayImage(img);
end

% 显示图像
function displayImage(img)
    ax = findobj('Tag', 'ImageAxes');
    axes(ax);
    imshow(img);
end

% 灰度直方图 & 均衡化
function histogramCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);
    figure;
    subplot(2, 2, 1); imshow(grayImg); title('灰度图');
    subplot(2, 2, 2); imhist(grayImg); title('直方图');
    equalizedImg = histeq(grayImg);
    subplot(2, 2, 3); imshow(equalizedImg); title('均衡化图像');
    subplot(2, 2, 4); imhist(equalizedImg); title('均衡化直方图');
end

% 对比度增强
function contrastEnhancementCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);
    % 线性变换
    linearImg = imadjust(grayImg);
    % 对数变换
    logImg = im2uint8(mat2gray(log(1 + double(grayImg))));
    % 指数变换
    expImg = im2uint8(mat2gray(exp(double(grayImg) / 255)));

    figure;
    subplot(2, 2, 1); imshow(grayImg); title('原始图像');
    subplot(2, 2, 2); imshow(linearImg); title('线性变换');
    subplot(2, 2, 3); imshow(logImg); title('对数变换');
    subplot(2, 2, 4); imshow(expImg); title('指数变换');
end

% 几何变换
function geometricTransformCallback(~, ~)
    img = evalin('base', 'img');
 
    % 缩放
    scaledImg = imresize( img, 0.5);
    % 旋转
    rotatedImg = imrotate( img, 45);

    figure;
    subplot(1, 3, 1); imshow(img); title('原始图像');
    subplot(1, 3, 2); imshow(scaledImg); title('缩放图像');
    subplot(1, 3, 3); imshow(rotatedImg); title('旋转图像');
end

% 去噪处理
function denoiseCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);
    % 添加噪声
    noisyImg = imnoise(grayImg, 'salt & pepper', 0.05);

    % 空域均值滤波
    meanFilteredImg = medfilt2(noisyImg);

    % 频域滤波
    fftImg = fft2(double(noisyImg));
    fftImgShift = fftshift(fftImg);
    H = fspecial('gaussian', size(fftImgShift), 10);
    freqFilteredImg = real(ifft2(ifftshift(fftImgShift .* H)));

    figure;
    subplot(2, 2, 1); imshow(grayImg); title('原始图像');
    subplot(2, 2, 2); imshow(noisyImg); title('加噪图像');
    subplot(2, 2, 3); imshow(meanFilteredImg); title('均值滤波');
    subplot(2, 2, 4); imshow(freqFilteredImg, []); title('频域滤波');
end


% 边缘检测
function edgeDetectionCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);

    % 使用不同算子进行边缘检测
    robertsEdge = edge(grayImg, 'roberts');
    prewittEdge = edge(grayImg, 'prewitt');
    sobelEdge = edge(grayImg, 'sobel');
    laplacianEdge = imfilter(grayImg, fspecial('laplacian'));

    figure;
    subplot(2, 2, 1); imshow(robertsEdge); title('Roberts 算子');
    subplot(2, 2, 2); imshow(prewittEdge); title('Prewitt 算子');
    subplot(2, 2, 3); imshow(sobelEdge); title('Sobel 算子');
    subplot(2, 2, 4); imshow(laplacianEdge, []); title('Laplacian 算子');
end

% 目标提取
function objectExtractionCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);

    % 简单阈值分割
    threshold = graythresh(grayImg);
    binaryImg = imbinarize(grayImg, threshold);

    figure;
    subplot(1, 2, 1); imshow(img); title('原始图像');
    subplot(1, 2, 2); imshow(binaryImg); title('提取目标');
end


% 特征提取
function featureExtractionCallback(~, ~)
    try
        img = evalin('base', 'img'); 
        grayImg = rgb2gray(img); 

        if size(grayImg, 1) < 32 || size(grayImg, 2) < 32
            errordlg('图像尺寸过小，请选择至少 32x32 的图像进行特征提取。', '错误');
            return;
        end

        % LBP 特征提取
        lbpFeatures = extractLBPFeatures(grayImg);

        % HOG 特征提取
        [hogFeatures, visualization] = extractHOGFeatures(grayImg);

        % 显示特征
        figure;
        subplot(1, 2, 1); bar(lbpFeatures); title('LBP 特征'); % 显示 LBP 特征
        subplot(1, 2, 2); plot(visualization); title('HOG 特征'); % 显示 HOG 特征可视化
    catch ME
        errordlg(['特征提取失败：', ME.message], '错误');
    end
end