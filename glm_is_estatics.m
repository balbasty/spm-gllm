function ok = glm_is_estatics(D)
% Checks whether a design matrix has the ESTATICS layout
%
% FORMAT glm_is_estatics(D)
%
% D - (N x K) Design matrix
%__________________________________________________________________________

% Yael Balbastre

K = size(D, 2);
T = eye(K,"logical");
T(:,end) = true;
T(end,:) = true;
DD = D' * D;
if ~any(DD .* ~T)
    ok = true;
else
    ok = false;
end
end