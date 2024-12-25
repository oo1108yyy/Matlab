%%第二版

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
    %灰度化
    R = img(:, :, 1);   % 提取颜色通道
    G = img(:, :, 2);
    B = img(:, :, 3);
    grayImg=0.299*R+0.587*G+0.114*B;   %利用公式进行灰度化

    % 创建GUI窗口
    hFig = figure('Name', 'Histogram & Equalization', 'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', ...
                  'Position', [100, 100, 250, 100]);
    
    % 显示灰度图和直方图按钮
    uicontrol('Style', 'pushbutton', 'String', '显示灰度图和直方图', ...
              'Position', [20 50 100 30], ...
              'Callback', @showHistogramCallback);
    
    % 直方图均衡化按钮
    uicontrol('Style', 'pushbutton', 'String', '直方图均衡化', ...
              'Position', [120 50 100 30], ...
              'Callback', @equalizeHistogramCallback);
    
    % 显示灰度图和直方图回调函数
    function showHistogramCallback(~, ~)
        figure;
        subplot(2, 1, 1); imshow(grayImg); title('灰度图');
        subplot(2, 1, 2); imhist(grayImg); title('直方图');
    end
    
    % 直方图均衡化回调函数
    function equalizeHistogramCallback(~, ~)
        equalizedImg = histeq(grayImg);
        figure;
        subplot(2, 1, 1); imshow(equalizedImg); title('均衡化图像');
        subplot(2, 1, 2); imhist(equalizedImg); title('均衡化直方图');
    end
end


%对比度增强
function contrastEnhancementCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);
    
    % 创建GUI窗口
    hFig = figure('Name', 'Contrast Enhancement',  'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', ...
                  'Position', [100, 100, 350, 100]);
    
    % 线性变换按钮
    uicontrol('Style', 'pushbutton', 'String', '线性变换', ...
              'Position', [20 50 100 30], ...
              'Callback', @linearTransformCallback);
    
    % 对数变换按钮
    uicontrol('Style', 'pushbutton', 'String', '对数变换', ...
              'Position', [120 50 100 30], ...
              'Callback', @logTransformCallback);
    
    % 指数变换按钮
    uicontrol('Style', 'pushbutton', 'String', '指数变换', ...
              'Position', [220 50 100 30], ...
              'Callback', @expTransformCallback);
    
    % 线性变换回调函数
    function linearTransformCallback(~, ~)
        linearImg = imadjust(grayImg);
        figure;
        imshow(linearImg);
        title( '线性变换');
    end
    
    % 对数变换回调函数
    function logTransformCallback(~, ~)
        logImg = im2uint8(mat2gray(log(1 + double(grayImg))));
        figure;
        imshow(logImg);
        title( '对数变换');

    end
    
    % 指数变换回调函数
    function expTransformCallback(~, ~)
        expImg = im2uint8(mat2gray(exp(double(grayImg) / 255)));
        figure;
        imshow(expImg);
        title( '指数变换');
    end
end



%缩放&旋转
function geometricTransformCallback(~, ~)
    img = evalin('base', 'img');
    
    % 创建GUI窗口
    hFig = figure('Name', 'Geometric Transformations', 'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', ...
                  'Position', [100, 100, 350, 100]);
    
    % 缩放按钮
    uicontrol('Style', 'pushbutton', 'String', '缩放', ...
              'Position', [20 50 100 30], ...
              'Callback', @scaleTransformCallback);
    
    % 旋转按钮
    uicontrol('Style', 'pushbutton', 'String', '旋转', ...
              'Position', [120 50 100 30], ...
              'Callback', @rotateTransformCallback);
    
    % 缩放变换回调函数
    function scaleTransformCallback(~, ~)
        prompt = {'请输入缩放比例:'};
        dlgtitle = '缩放图像';
        scaleRatio = inputdlg(prompt, dlgtitle);
        if isequal(scaleRatio, {}) || isempty(scaleRatio{1})
            return; % 如果用户取消或没有输入，则返回
        end
        scaleRatio = str2double(scaleRatio{1});
        if isnan(scaleRatio)
            error('输入的不是有效的数字');
        end
        scaledImg = imresize(img, scaleRatio);
        figure;
        imshow(scaledImg);
        title('缩放图像');
    end
    
    % 旋转变换回调函数
    function rotateTransformCallback(~, ~)
        prompt = {'请输入旋转角度(度):'};
        dlgtitle = '旋转图像';
        angle = inputdlg(prompt, dlgtitle);
        if isequal(angle, {}) || isempty(angle{1})
            return; % 如果用户取消或没有输入，则返回
        end
        angle = str2double(angle{1});
        if isnan(angle)
            error('输入的不是有效的数字');
        end
        rotatedImg = imrotate(img, angle);
        figure;
        imshow(rotatedImg);
        title('旋转图像');
    end
end




% 去噪处理
function denoiseCallback(~, ~)
 img = evalin('base', 'img');
    grayImg = rgb2gray(img);
    
    % 弹出窗口选择噪声类型及输入参数
    [noiseType, noiseParam] = selectNoiseGUI();
    
    % 根据选择的噪声类型和参数添加噪声
    switch noiseType
        case 'Salt & Pepper'
            noisyImg = imnoise(grayImg, 'salt & pepper', noiseParam);
        case 'Gaussian'
            noisyImg = imnoise(grayImg, 'gaussian', 0, noiseParam^2);        
        otherwise
            error('Unsupported noise type');
    end
    
    % 显示加噪图像
    figure;
    imshow(noisyImg);
    title(['加噪图像 (' noiseType ')']);
    
    % 创建GUI窗口
    hFig = figure('Name', 'Denoise Processing', 'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', ...
                  'Position', [100, 100, 350, 100]);
    
    % 空域均值滤波按钮
    uicontrol('Style', 'pushbutton', 'String', '空域均值滤波', ...
              'Position', [20 50 100 30], ...
              'Callback', @meanFilterCallback);
    
    % 频域滤波按钮
    uicontrol('Style', 'pushbutton', 'String', '频域滤波', ...
              'Position', [120 50 100 30], ...
              'Callback', @freqFilterCallback);
    
    % 空域均值滤波回调函数
    function meanFilterCallback(~, ~)
        meanFilteredImg = medfilt2(noisyImg);
        figure;
        imshow(meanFilteredImg);
        title('空域均值滤波');
    end
    
    % 频域滤波回调函数
    function freqFilterCallback(~, ~)
        fftImg = fft2(double(noisyImg));
        fftImgShift = fftshift(fftImg);
        H = fspecial('gaussian', size(fftImgShift), 10);
        freqFilteredImg = real(ifft2(ifftshift(fftImgShift .* H)));
        figure;
        imshow(freqFilteredImg, []);
        title('频域滤波');
    end
end

% 噪声选择GUI函数
function [selectedType, selectedParam] = selectNoiseGUI()
    % 定义噪声类型列表
    noiseTypeList = {'Salt & Pepper', 'Gaussian'};
    
    % 弹出对话框选择噪声类型
    disp('请选择噪声类型（输入数字）:');
    disp('1 - Salt & Pepper');
    disp('2 - Gaussian');
    selectedTypeIdx = input('请输入您的选择（1 或 2）: ');
    
    % 检查输入是否有效
    if isnan(selectedTypeIdx) || selectedTypeIdx < 1 || selectedTypeIdx > length(noiseTypeList)
        error('无效的噪声类型选择。');
    end
    selectedType = noiseTypeList{selectedTypeIdx};
    
    % 弹出对话框输入噪声参数
     if strcmp(selectedType, 'Salt & Pepper')
        selectedParam = input('请输入Salt & Pepper噪声的参数（0到1之间的值）: ');
    else
        selectedParam = input('请输入Gaussian噪声的标准差（正数）: ');
    end
    
    % 检查输入是否有效
    if isnan(selectedParam) || selectedParam < 0
        error('无效的噪声参数输入。');
    end
end



% 边缘检测
function edgeDetectionCallback(~, ~)
    img = evalin('base', 'img');
    grayImg = rgb2gray(img);

    % 创建GUI窗口
    hFig = figure('Name', 'Edge Detection', 'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', ...
                  'Position', [100, 100, 450, 100]);
    
    % Roberts 算子按钮
    uicontrol('Style', 'pushbutton', 'String', 'Roberts 算子', ...
              'Position', [20 50 100 30], ...
              'Callback', @robertsEdgeCallback);
    
    % Prewitt 算子按钮
    uicontrol('Style', 'pushbutton', 'String', 'Prewitt 算子', ...
              'Position', [120 50 100 30], ...
              'Callback', @prewittEdgeCallback);
    
    % Sobel 算子按钮
    uicontrol('Style', 'pushbutton', 'String', 'Sobel 算子', ...
              'Position', [220 50 100 30], ...
              'Callback', @sobelEdgeCallback);
    
    % Laplacian 算子按钮
    uicontrol('Style', 'pushbutton', 'String', 'Laplacian 算子', ...
              'Position', [320 50 100 30], ...
              'Callback', @laplacianEdgeCallback);
    
    % Roberts 算子回调函数
    function robertsEdgeCallback(~, ~)
        robertsEdge = edge(grayImg, 'roberts');
        figure;
        imshow(robertsEdge);
        title('Roberts 算子');
    end
    
    % Prewitt 算子回调函数
    function prewittEdgeCallback(~, ~)
        prewittEdge = edge(grayImg, 'prewitt');
        figure;
        imshow(prewittEdge);
        title('Prewitt 算子');
    end
    
    % Sobel 算子回调函数
    function sobelEdgeCallback(~, ~)
        sobelEdge = edge(grayImg, 'sobel');
        figure;
        imshow(sobelEdge);
        title('Sobel 算子');
    end
    
    % Laplacian 算子回调函数
    function laplacianEdgeCallback(~, ~)
        laplacianEdge = imfilter(grayImg, fspecial('laplacian'));
        figure;
        imshow(laplacianEdge, []);
        title('Laplacian 算子');
    end
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
        img = evalin('base', 'img'); 
        grayImg = rgb2gray(img); 

        if size(grayImg, 1) < 32 || size(grayImg, 2) < 32
            errordlg('图像尺寸过小，请选择至少 32x32 的图像进行特征提取。', '错误');
            return;
        end

        % 创建GUI窗口
        hFig = figure('Name', 'Feature Extraction', 'NumberTitle', 'off', ...
                      'MenuBar', 'none', 'ToolBar', 'none', ...
                      'Position', [100, 100, 300, 100]);
        
        % LBP 特征提取按钮
        uicontrol('Style', 'pushbutton', 'String', 'LBP 特征提取', ...
                  'Position', [20 60 130 30], ...
                  'Callback', @lbpFeatureCallback);
        
        % HOG 特征提取按钮
        uicontrol('Style', 'pushbutton', 'String', 'HOG 特征提取', ...
                  'Position', [160 60 130 30], ...
                  'Callback', @hogFeatureCallback);
   
        % LBP 特征提取回调函数
        function lbpFeatureCallback(~, ~)
            lbpFeatures = extractLBPFeatures(grayImg);
            figure;
            bar(lbpFeatures);
            title('LBP 特征');
        end
        
        % HOG 特征提取回调函数
        function hogFeatureCallback(~, ~)
            [hogFeatures, visualization] = extractHOGFeatures(grayImg);
            figure;
            plot(visualization);
            title('HOG 特征');
        end
end

