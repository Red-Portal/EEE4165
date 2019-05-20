box1 = ones(11,11) / (11.0 * 11.0);
box2 = ones(21,21) / (21.0 * 21.0);

gauss1 = gaussian_kernel(31, 5);
gauss2 = gaussian_kernel(55, 9);

chirp     = load_image('Linear_chirp.tif');
imshow(chirp)
[box1_result, box1_mag] = fourier_filter(chirp, box1, [512 512]);
imagesc(mag2db(box1_mag));
saveas(gcf, "output/impl2_box1_spec.png")
close(gcf)

box1_result = quantize_image(abs(box1_result));
imshow(box1_result(1:500,1:500));
saveas(gcf, "output/impl2_box1_result.png")
close(gcf)

[box2_result, box2_mag] = fourier_filter(chirp, box2, [512 512]);
imagesc(mag2db(box2_mag));
saveas(gcf, "output/impl2_box2_spec.png")
close(gcf)

box2_result = quantize_image(abs(box2_result));
imshow(box2_result);
imshow(box2_result(1:500,1:500));
saveas(gcf, "output/impl2_box2_result.png")
close(gcf)

[gauss1_result, gauss1_mag] = fourier_filter(chirp, gauss1, [512 512]);
imagesc(mag2db(gauss1_mag));
saveas(gcf, "output/impl2_gauss1_spec.png")
close(gcf)

gauss1_result = quantize_image(abs(gauss1_result));
imshow(gauss1_result(1:500,1:500));
saveas(gcf, "output/impl2_gauss1_result.png")
close(gcf)

[gauss2_result, gauss2_mag] = fourier_filter(chirp, gauss2, [512 512]);
imagesc(mag2db(gauss2_mag));
saveas(gcf, "output/impl2_gauss2_spec.png")
close(gcf)

gauss2_result = quantize_image(abs(gauss2_result));
imshow(gauss2_result(1:500, 1:500));
saveas(gcf, "output/impl2_gauss2_result.png")
close(gcf)
