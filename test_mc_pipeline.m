folder.root = '/Users/balbasty/Dropbox/Workspace/data/hMRI/hmri_sample_dataset_with_maps_coreg';
folder.out  = '/Users/balbasty/Dropbox/Workspace/data/hMRI/mc';
folder.PDw  = 'pdw_mfc_3dflash_v1i_R4_0009_pool3';
folder.T1w  = 't1w_mfc_3dflash_v1i_R4_0015_pool3';
folder.MTw  = 'mtw_mfc_3dflash_v1i_R4_0012_pool3';

% -------------------------------------------------------------------------
% Get files
fnames.PDw = spm_select('FPList',[folder.root '/' folder.PDw '/'],'.nii$');
fnames.T1w = spm_select('FPList',[folder.root '/' folder.T1w '/'],'.nii$');
fnames.MTw = spm_select('FPList',[folder.root '/' folder.MTw '/'],'.nii$');

npdw = size(fnames.PDw,1);
nt1w = size(fnames.T1w,1);
nmtw = size(fnames.MTw,1);

% -------------------------------------------------------------------------
% Build virtual array of observations
Y = spm_volarray([fnames.PDw;fnames.T1w;fnames.MTw]);

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
X = [zeros(numel(TE),3) -TE'];
X(          1:     npdw,     1:end-1) = X(          1:     npdw,     1:end-1) + [1 0 0];
X(     npdw+1:nt1w+npdw,     1:end-1) = X(     npdw+1:nt1w+npdw,     1:end-1) + [0 1 0];
X(nt1w+npdw+1:nt1w+npdw+nmtw,1:end-1) = X(nt1w+npdw+1:nt1w+npdw+nmtw,1:end-1) + [0 0 1];

% tmp = X;
% X = zeros(numel(TE),4,2,2);
% X(:,:,1,1) = tmp;
% X(:,:,2,2) = tmp;
% X = reshape(X, [], 4*2,2);

M = size(X,1);
K = size(X,2);
C = size(X,3);

% -------------------------------------------------------------------------
% Run fit
B = zeros([size(Y,1) size(Y,2) size(Y,3) 4 2]);
R = zeros([size(Y,1) size(Y,2) size(Y,3)]);
zz = round(2*size(Y,3)/3);
for z=zz
    Y1           = reshape(Y(:,:,z,:), [], M);
    [B1,~,R1]    = gllm_fit(Y1,X,1,struct('verb',2,'mc',2,'mode','sym','proc',@proc));
    msk          = B1(:,end,1) > B1(:,end,2);
    B1(msk,:,:)  = B1(msk,:,end:-1:1);
    B(:,:,z,:,:) = reshape(B1, [size(Y,1) size(Y,2) 1 4 2]);
    R(:,:,z)     = reshape(R1, [size(Y,1) size(Y,2)]);
end

% -------------------------------------------------------------------------
% Save
ninp               = nifti(fnames.PDw(1,:));

nout               = ninp;
nout.dat.fname     = [folder.out '/RMSE.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = sqrt(R);

for c=1:2

nout               = ninp;
nout.dat.fname     = [folder.out '/PDw_c' num2str(c) '.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = exp(B(:,:,:,1,c));

nout               = ninp;
nout.dat.fname     = [folder.out '/T1w_c' num2str(c) '.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = exp(B(:,:,:,2,c));

nout               = ninp;
nout.dat.fname     = [folder.out '/MTw_c' num2str(c) '.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = exp(B(:,:,:,3,c));

nout               = ninp;
nout.dat.fname     = [folder.out '/R2star_c' num2str(c) '.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = B(:,:,:,end,c) * 1000;

end

% -------------------------------------------------------------------------
% Plot
z = zz;
figure

subplot(2,4,1)
imagesc(exp(B(:,:,z,1,1)), [0,1000]);
subplot(2,4,2)
imagesc(exp(B(:,:,z,2,1)), [0,1000]);
subplot(2,4,3)
imagesc(exp(B(:,:,z,3,1)), [0,1000]);
subplot(2,4,4)
imagesc(B(:,:,z,4,1)*1000, [0 100]);

subplot(2,4,5)
imagesc(exp(B(:,:,z,1,2)), [0,1000]);
subplot(2,4,6)
imagesc(exp(B(:,:,z,2,2)), [0,1000]);
subplot(2,4,7)
imagesc(exp(B(:,:,z,3,2)), [0,1000]);
subplot(2,4,8)
imagesc(B(:,:,z,4,2)*1000, [0 100]);

function B = proc(B)
B(:,1:end-1,:) = min(max(B(:,1:end-1,:),-32),32);
B(:,end,:)     = max(B(:,end,:),0);
end