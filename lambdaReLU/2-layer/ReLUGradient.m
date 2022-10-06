function g = ReLUGradient(z)
  g = max((z>=0), 0.05);
endfunction