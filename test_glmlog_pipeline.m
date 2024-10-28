FOLDER  = '/Users/balbasty/localdata/antoine/ExampleDataITS';
VARIANT = 'ITS';   % (ITS|Standard)
REP     = 'rep1';       % (rep1|rep2|rep3)
DROP    = 3;

% -------------------------------------------------------------------------
% Get files
fnames = spm_select('FPList',[FOLDER '/' VARIANT '/' REP '/'],'.nii$');

% Drop first few echoes
fnames = fnames(DROP+1:end,:);

% -------------------------------------------------------------------------
% Build virtual array of observations
Y = spm_volarray(fnames);

% -------------------------------------------------------------------------
% Parse TE
warning('off','MATLAB:structOnObject');
TE = [];
pattern = 'TR=(?<TR>.+)ms/TE=(?<TE>.+)ms/FA=(?<FA>.+deg)/';
for i=1:numel(struct(Y).vol)
    vol1 = struct(Y).vol(i);
    meta = regexp(vol1.descrip, pattern, 'names');
    TE = [TE str2num(meta.TE)]; 
end

% -------------------------------------------------------------------------
% Build design matrix and covariance basis
X = (-TE') .^ (0:1);
Q =  (TE') .^ (0:1);

M = size(X,1);
K = size(X,2);

% Make sparse basis for spm_reml
Q = {spdiags(Q(:,1),0,M,M) spdiags(Q(:,2),0,M,M)};

% -------------------------------------------------------------------------
% ReML estimate of covariance

% -------------------------------
% Estimate mask of voxels to keep
% > To speed things up, use best 2^16 voxels (approx.)
MSK = gllm_reml_mask(Y,X,2^16);

% --------------------------------
% Collect data in the mask of ReML
N   = sum(MSK(:));
YM  = zeros(N,M);
off = 0;
for z=1:size(Y,3)
    Y1  = reshape(Y(:,:,z,:),[],M);
    M1  = reshape(MSK(:,:,z), [], 1);
    Y1  = reshape(Y1(M1,:,:), [], M);
    chk = size(Y1,1);
    YM(off+1:off+chk,:) = Y1;
    off = off + chk;
end

% ------------------------------------
% Take log
YM = log(YM + 1e-4);

% ------------------------------------
% Prepare observed covariance for ReML
B  = YM / X';
R  = B*X' - YM;
S  = dot(R,R,2) / (M-K);
RR = (R' * (R ./ S)) / N;
YY = YM ./ sqrt(S);
YY = (YY' * YY) / N;

% -------------------------------------------------------------------------
% Estimate covariance
[C,h] = spm_reml(YY,X,Q);

% -------------------------------------------------------------------------
% Look at fits
figure
idx = 1;
hold off
scatter(TE, YM(idx,:));
hold on
T = linspace(0,18,128); 
plot(T, B(idx,:)*((-T)' .^ (0:1))'); 
hold off;

% -------------------------------------------------------------------------
% Run complete fit
if strcmpi(FIT(1:2), 'nl'), fit = @gllm_fit;
else,                       fit = @gllm_logfit; end

B = zeros([size(Y,1) size(Y,2) size(Y,3) K]);
for z=1:size(Y,3)
    Y1 = reshape(Y(:,:,z,:), [], M);
    B1 = fit(Y1,X,1./C,struct('verb',1));
    B(:,:,z,:) = reshape(B1, [size(Y,1) size(Y,2) 1 K]);
end

% -------------------------------------------------------------------------
% Save
ninp               = nifti(fnames(1,:));

nout               = ninp;
nout.dat.fname     = [FOLDER '/' FIT '_' VARIANT '_' REP '_S.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = exp(B(:,:,:,1));

nout               = ninp;
nout.dat.fname     = [FOLDER '/' FIT '_' VARIANT '_' REP '_R2star.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = B(:,:,:,2) * 1000;

% -------------------------------------------------------------------------
% Plot
figure
subplot(1,3,1)
imagesc(exp(B(:,:,64,1)));
subplot(1,3,2)
imagesc(B(:,:,64,2)*1000, [0 100]);
subplot(1,3,3)
bar(C)
