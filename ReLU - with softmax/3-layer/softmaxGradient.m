function g = softmaxGradient(x)
  g = softmax(x) * eye(length(x)) - softmax(x)' * softmax(x);
endfunction