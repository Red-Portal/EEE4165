function y = center_freq(x)
    img_dim = size(x);
    [X, Y]  = meshgrid(0:img_dim(2)-1, 0:img_dim(1)-1);
    shifted = ((-1.0).^(X+Y)).*x;
    y       = shifted;
end