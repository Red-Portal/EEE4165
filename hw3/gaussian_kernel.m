function k = gaussian_kernel(dim, sigma)
    coor   = -floor(dim/2):floor(dim/2);
    [X, Y] = meshgrid(coor, coor);
    coeff  = exp(-(X.^2 + Y.^2) / (2*sigma*sigma));
    k      = coeff / sum(coeff(:));
end