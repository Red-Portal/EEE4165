
astronaut      = load_image('Astronaut.tif');
N              = 1024;
[spectrum, ~]  = complex_image(astronaut, [N N]);

imagesc(mag2db(spectrum));

saveas(gcf, "output/impl3_spectrum.png")
close(gcf)

mask = ones([N N]);
mask(:,476:485) = 0;
mask(:,542:549) = 0;

centered    =  center_freq(astronaut);
cplex_image = fft2(centered, N, N);
cplex_recon = f_astro .* mask;

imagesc(mag2db(abs(cplex_recon)));
saveas(gcf, "output/impl3_masked.png")
close(gcf)

astro_recon = ifft2(cplex_recon);
origin_dim  = size(astronaut);
astro_recon = astro_recon(1:origin_dim(1),1:origin_dim(2));
astro_recon = quantize_image(abs(astro_recon));
imshow(astro_recon);
saveas(gcf, "output/impl3_recon.png")
close(gcf)
