building = load_image('Building.tif');
[building_magni, building_phase] = complex_image(building, [1024 1024]);
imagesc(mag2db(building_magni));
saveas(gcf, "output/impl1_building_magnitude.png")
close(gcf)

imagesc(mag2db(building_phase));
saveas(gcf, "output/impl1_building_phase.png")
close(gcf)

rec = load_image('Rectangle.tif');
[rec_magni, rec_phase] = complex_image(rec, [1024 1024]);
imagesc(mag2db(rec_magni));
saveas(gcf, "output/impl1_rec_magnitude.png")
close(gcf)

imagesc(mag2db(rec_phase));
saveas(gcf, "output/impl1_rec_phase.png")
close(gcf)

% reconstruct using spectrum of Building.tif and phase of Rectangle.tif
compl1  = building_magni.*exp(1.0j*rec_phase);
recon1  = center_freq(ifft2(compl1));
impl1_1 = quantize_image(abs(recon1));

% reconstruct using phase of Building.tif and spectrum of Rectangle.tif
compl2  = rec_magni.*exp(1.0j*building_phase);
recon2  = center_freq(ifft2(compl2));
impl1_2 = quantize_image(abs(recon2));

rec_dim = size(rec);
imshow(impl1_1(1:rec_dim(1),1:rec_dim(2)));
saveas(gcf, "output/impl1_1.png")
close(gcf)

building_dim = size(building);
imshow(impl1_2(1:building_dim(1),1:building_dim(2)));
saveas(gcf, "output/impl1_2.png")
close(gcf)
