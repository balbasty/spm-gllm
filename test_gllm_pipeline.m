FOLDER = '/Users/balbasty/localdata/antoine/ExampleDataITS';

% -------------------------------------------------------------------------
% Get files
fnames = spm_select('FPList',[FOLDER '/Standard/rep1/'],'.nii$');

% -------------------------------------------------------------------------
% Build virtual array of observations
Y = spm_volarray(fnames);

% -------------------------------------------------------------------------
% Parse TE
TE = [];
pattern = 'TR=(?<TR>.+)ms/TE=(?<TE>.+)ms/FA=(?<FA>.+deg)/';
for i=1:numel(struct(A).vol)
    vol1 = struct(A).vol(i);
    meta = regexp(vol1.descrip, pattern, 'names');
    TE = [TE str2num(meta.TE)]; 
end

% -------------------------------------------------------------------------
% Build design matrix and covariance basis
X = [ones(numel(TE),1) -TE'];
Q = [ones(numel(TE),1)  TE']';

M = size(X,1);
K = size(X,2);

% -------------------------------------------------------------------------
% Estimate mask of voxels to keep
MSK = gllm_reml_mask(A,X);

% -------------------------------------------------------------------------
% Collect data in the mask of ReML
YM  = zeros(numel(MSK),size(Y,4));
off = 0;
for z=1:size(Y,3)
    Y1  = reshape(Y(:,:,z,:),[],M);
    M1  = reshape(MSK(:,:,z), [], 1);
    Y1  = reshape(Y1(M1,:,:), [], M);
    chk = size(Y1,1);
    YM(off+1:off+chk,:) = Y1;
    off = off + chk;
end

% -------------------------------------------------------------------------
% Estimate covariance
C = gllm_reml(YM,X,Q,struct('verb',2));

% -------------------------------------------------------------------------
% Run complete fit
B = zeros([size(Y,1) size(Y,2) size(Y,3) K]);
for z=1:size(Y,3)
    Y1 = reshape(Y(:,:,z,:), [], M);
    B1 = gllm_fit(Y1,X,1./C,struct('verb',1));
    B(:,:,z,:) = reshape(B1, [size(Y,1) size(Y,2) 1 K]);
end

% -------------------------------------------------------------------------
% Plot
figure
subplot(1,3,1)
imagesc(exp(B(:,:,64,1)));
subplot(1,3,2)
imagesc(B(:,:,64,2), [0 .1]);
subplot(1,3,3)
bar(C)

