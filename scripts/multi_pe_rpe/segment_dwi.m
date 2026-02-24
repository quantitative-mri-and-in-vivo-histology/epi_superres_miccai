% Segment DWI using SPM12 New Segment with eTPM — multi-channel
%
% Channel 1: mean b=0  — bias correction enabled  (biasreg = 0.001)
% Channel 2: DTI FA    — bias correction suppressed (biasreg = 10, max GUI value)
%
% Loops over all resolutions (native 1.7mm + downsampled 2.0/2.5/3.0/3.4mm)
% and both processing modes (b0_rpe, full_rpe).
%
% Input files expected in each dwi/ directory (produced by process_dwi.py):
%   dwi_lr_preprocessed_mean_b0.nii.gz      (b0_rpe)
%   dwi_lr_preprocessed_dti_fa.nii.gz       (b0_rpe)
%   dwi_merged_preprocessed_mean_b0.nii.gz  (full_rpe)
%   dwi_merged_preprocessed_dti_fa.nii.gz   (full_rpe)

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(fileparts(script_dir));
tpm        = fullfile(root_dir, 'data', 'templates', 'eTPM.nii');

% Resolutions: {dwi_dir, display_name}
resolutions = {
    fullfile(root_dir, 'data', 'multi_pe_rpe', 'native_res',        'processed', 'dwi'), 'native 1.7mm';
    fullfile(root_dir, 'data', 'multi_pe_rpe', 'downsampled_2p0mm', 'processed', 'dwi'), 'downsampled 2.0mm';
    fullfile(root_dir, 'data', 'multi_pe_rpe', 'downsampled_2p5mm', 'processed', 'dwi'), 'downsampled 2.5mm';
    fullfile(root_dir, 'data', 'multi_pe_rpe', 'downsampled_3p0mm', 'processed', 'dwi'), 'downsampled 3.0mm';
    fullfile(root_dir, 'data', 'multi_pe_rpe', 'downsampled_3p4mm', 'processed', 'dwi'), 'downsampled 3.4mm';
};

% Modes: {mode_name, preprocessed_file_prefix}
modes = {
    'b0_rpe',   'dwi_lr_preprocessed';
    'full_rpe', 'dwi_merged_preprocessed';
};

spm('defaults', 'FMRI');

for r = 1:size(resolutions, 1)
    dwi_dir  = resolutions{r, 1};
    res_name = resolutions{r, 2};

    if ~exist(dwi_dir, 'dir')
        fprintf('Skipping %s (directory not found)\n', res_name);
        continue;
    end

    fprintf('\n=== %s ===\n', res_name);

    for m = 1:size(modes, 1)
        mode_name = modes{m, 1};
        prefix    = modes{m, 2};

        fprintf('  Mode: %s\n', mode_name);

        mean_b0_gz  = fullfile(dwi_dir, [prefix '_mean_b0.nii.gz']);
        dti_fa_gz   = fullfile(dwi_dir, [prefix '_dti_fa.nii.gz']);
        mean_b0_nii = fullfile(dwi_dir, [prefix '_mean_b0.nii']);
        dti_fa_nii  = fullfile(dwi_dir, [prefix '_dti_fa.nii']);

        if ~exist(mean_b0_gz, 'file') && ~exist(mean_b0_nii, 'file')
            fprintf('    mean_b0 not found, skipping\n');
            continue;
        end
        if ~exist(dti_fa_gz, 'file') && ~exist(dti_fa_nii, 'file')
            fprintf('    dti_fa not found, skipping\n');
            continue;
        end

        % SPM12 requires uncompressed NIfTI
        if ~exist(mean_b0_nii, 'file')
            fprintf('    Uncompressing mean_b0...\n');
            gunzip(mean_b0_gz, dwi_dir);
        end
        if ~exist(dti_fa_nii, 'file')
            fprintf('    Uncompressing dti_fa...\n');
            gunzip(dti_fa_gz, dwi_dir);
        end

        matlabbatch = {};

        % Channel 1: mean b=0 — B1 bias correction enabled
        matlabbatch{1}.spm.spatial.preproc.channel(1).vols     = {[mean_b0_nii, ',1']};
        matlabbatch{1}.spm.spatial.preproc.channel(1).biasreg  = 0.001;
        matlabbatch{1}.spm.spatial.preproc.channel(1).biasfwhm = 60;
        matlabbatch{1}.spm.spatial.preproc.channel(1).write    = [0 0];

        % Channel 2: DTI FA — bias correction suppressed (biasreg = 10 = max SPM GUI value)
        matlabbatch{1}.spm.spatial.preproc.channel(2).vols     = {[dti_fa_nii, ',1']};
        matlabbatch{1}.spm.spatial.preproc.channel(2).biasreg  = 10;
        matlabbatch{1}.spm.spatial.preproc.channel(2).biasfwhm = 60;
        matlabbatch{1}.spm.spatial.preproc.channel(2).write    = [0 0];

        % Tissue classes (6 classes, same as segment_mprage)
        matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {[tpm, ',1']};
        matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
        matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {[tpm, ',2']};
        matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
        matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {[tpm, ',3']};
        matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
        matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {[tpm, ',4']};
        matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
        matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {[tpm, ',5']};
        matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
        matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {[tpm, ',6']};
        matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
        matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
        matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];

        % Warping parameters (no deformation fields written)
        matlabbatch{1}.spm.spatial.preproc.warp.mrf     = 1;
        matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
        matlabbatch{1}.spm.spatial.preproc.warp.reg     = [0 0.001 0.5 0.05 0.2];
        matlabbatch{1}.spm.spatial.preproc.warp.affreg  = 'mni';
        matlabbatch{1}.spm.spatial.preproc.warp.fwhm    = 0;
        matlabbatch{1}.spm.spatial.preproc.warp.samp    = 3;
        matlabbatch{1}.spm.spatial.preproc.warp.write   = [0 0];
        matlabbatch{1}.spm.spatial.preproc.warp.vox     = NaN;
        matlabbatch{1}.spm.spatial.preproc.warp.bb      = [NaN NaN NaN
                                                            NaN NaN NaN];

        fprintf('    Running segmentation...\n');
        spm_jobman('run', matlabbatch);
        fprintf('    Done.\n');
    end
end

fprintf('\nAll segmentations complete.\n');
