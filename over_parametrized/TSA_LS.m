function [acc_vec_LS] = TSA_LS (x, A3, b3)
% compute the test set accuracy
b_test1 = norm(A3*x - b3);  
acc_vec_LS = b_test1;
end