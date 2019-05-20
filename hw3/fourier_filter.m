function [y, filt_spec] = fourier_filter(img, filter, pad_dim)    
    img_dim       = size(img);
    filter_dim    = size(filter);
    
    % zero pad image, filter
    filter_padded = zeros(pad_dim);
    image_padded  = zeros(pad_dim);
    image_padded(1:img_dim(1),1:img_dim(2)) = img(:,:);
    filter_padded(1:filter_dim(1),1:filter_dim(2)) = filter(:,:);
    
    % obtain fourier domain representation
    image_shifted  = center_freq(image_padded);
    filter_shifted = center_freq(filter_padded);
    fimg           = fft2(image_shifted, pad_dim(1), pad_dim(2));
    ffilt          = fft2(filter_shifted, pad_dim(1), pad_dim(2));
    
    % apply filter in fourier domain
    result         = ifft2(fimg .* ffilt);
    
    y              = center_freq(result);
    filt_spec      = abs(ffilt);
end
