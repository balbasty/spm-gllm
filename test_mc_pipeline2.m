folder.inp = '/Users/balbasty/localdata/antoine/ExampleDataITS/Standard/rep1';
folder.out  = '/Users/balbasty/Dropbox/Workspace/data/gllmmc/';

% -------------------------------------------------------------------------
% Get files
fnames = spm_select('FPList',folder.inp,'.nii$');

% -------------------------------------------------------------------------
% Build virtual array of observations
Y = spm_volarray([fnames]);

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
X = [ones(numel(TE),1) -TE'];

M = size(X,1);
K = size(X,2);
C = size(X,3);

% -------------------------------------------------------------------------
% Run fit
B = zeros([size(Y,1) size(Y,2) size(Y,3) 2 2]);
R = zeros([size(Y,1) size(Y,2) size(Y,3)]);
R(:) = inf;
zz = round(2*size(Y,3)/3);
for z=zz
    for repeat=1:3
        Y1           = reshape(Y(:,:,z,:), [], M);
        [B1,~,R1]    = gllm_fit(Y1,X,1,struct('verb',2,'mc',2,'mode','sym','proc',@proc,'init',0,'iter',256));
        msk          = B1(:,end,1) > B1(:,end,2);
        B1(msk,:,:)  = B1(msk,:,end:-1:1);
        B1           = reshape(B1, [size(Y,1) size(Y,2) 1 2 2]);
        R1           = reshape(R1, [size(Y,1) size(Y,2)]);
        msk          = R1 < R(:,:,z);
        B(:,:,z,:,:) = B(:,:,z,:,:) .* (~msk) + B1 .* msk;
        Rz           = R(:,:,z);
        Rz(msk)      = R1(msk);
        R(:,:,z)     = Rz;
    end
end

% -------------------------------------------------------------------------
% Save
ninp               = nifti(fnames(1,:));

nout               = ninp;
nout.dat.fname     = [folder.out '/RMSE.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = sqrt(R);

for c=1:2

nout               = ninp;
nout.dat.fname     = [folder.out '/S_c' num2str(c) '.nii'];
nout.dat.dtype     = 'FLOAT32';
nout.dat.scl_slope = 1;
nout.dat.scl_inter = 0;
create(nout);
nout.dat(:,:,:)    = exp(B(:,:,:,1,c));

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

subplot(2,3,1)
imagesc(exp(B(:,:,z,1,1)), [0,1000]);
subplot(2,3,2)
imagesc(B(:,:,z,2,1)*1000, [0 100]);

subplot(2,3,4)
imagesc(exp(B(:,:,z,1,2)), [0,1000]);
subplot(2,3,5)
imagesc(B(:,:,z,2,2)*1000, [0 100]);

subplot(2,3,3)
imagesc(sqrt(R(:,:,z)), [0 10]);

function B = proc(B)
B(:,1:end-1,:) = min(max(B(:,1:end-1,:),-32),32);
B(:,end,:)     = max(B(:,end,:),0);
end