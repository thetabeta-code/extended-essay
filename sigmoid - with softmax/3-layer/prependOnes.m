function[result] = prependOnes(X)
  result = [ones(rows(X),1) , X];
end