function y = quantize_image(x)
  peak   = max(max(x));
  scaled = x / peak * 255.0;
  y = uint8(scaled);
end