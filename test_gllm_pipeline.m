FOLDER  = '/Users/balbasty/localdata/antoine/ExampleDataITS';
VARIANT = 'Standard';        % (ITS|Standard)
REP     = 'rep3';       % (rep1|rep2|rep3)
FIT     = 'nlreml';     % (nlreml|nlls|ols)
DROP    = 3;

% -------------------------------------------------------------------------
% Get files
fnames = spm_select('FPList',[FOLDER '/' VARIANT '/' REP '/'],'.nii$');

% Drop first few echoes
fnames = fnames(DROP+1:end,:);

% -------------------------------------------------------------------------
% Build virtual array of observations
Y = spm_volarray(fnames);

% Noisify data to account for discretization
Y.rand = true;

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
Q = [ones(numel(TE),1) TE']';

% X = spm_orth(X);

M = size(X,1);
K = size(X,2);

% -------------------------------------------------------------------------
% ReML estimate of covariance
if strcmpi(FIT, 'nlreml')
    % ---------------------------------------------------------------------
    % Estimate mask of voxels to keep
    % > To speed things up, use best 2^16 voxels (approx.)
    MSK = gllm_reml_mask(Y,X,2^16);
    
    % ---------------------------------------------------------------------
    % Collect data in the mask of ReML
    YM  = zeros(sum(MSK(:)),M);
    off = 0;
    for z=1:size(Y,3)
        Y1  = reshape(Y(:,:,z,:),[],M);
        M1  = reshape(MSK(:,:,z), [], 1);
        Y1  = reshape(Y1(M1,:,:), [], M);
        chk = size(Y1,1);
        YM(off+1:off+chk,:) = Y1;
        off = off + chk;
    end

    % ---------------------------------------------------------------------
    % Estimate covariance
    [C,~,h] = gllm_reml(YM,X,Q,struct('verb',2));
else
    C = 1;
end

% -------------------------------------------------------------------------
% Look at fits
Y0 = YM;

N  = size(Y0,1);
B  = gllm_fit(Y0,X,1./C);
R  = Y0 - exp(B*X');
S  = dot(R,R,2) / (M-K);
RR = (R' * (R ./ S)) / N;

figure
idx = 1;
hold off
scatter(TE, Y0(idx,:));
hold on
T = linspace(0,18,128); 
plot(T, exp(B(idx,:)*((-T)' .^ (0:1))')); 
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

