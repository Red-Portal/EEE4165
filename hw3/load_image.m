function y = load_image(fname)
    img = imread(fname);
    y   = double(img) / 255.0;
end