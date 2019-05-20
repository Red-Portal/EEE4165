function [magni, phase] = complex_image(img, pad_dim)
    img_dim = size(img);
    padded  = zeros(pad_dim);
    padded(1:img_dim(1),1:img_dim(2)) = img(:,:);
    shifted = center_freq(padded);
    fimg    = fft2(shifted, pad_dim(1), pad_dim(2));
    magni   = abs(fimg);
    phase   = angle(fimg);
end