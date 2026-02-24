% Segment MTsat using SPM12 New Segment with eTPM (NO bias correction)
% Runs segmentation only on native resolution MTsat
% Input: MTsat.nii files in each subject's mpm/ directory
% Output: c1, c2, c3 tissue probability maps (c2 = WM)

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(fileparts(script_dir));
tpm        = fullfile(root_dir, 'data', 'templates', 'eTPM.nii');

processed_dir = fullfile(root_dir, 'data', 'single_pe_rpe', 'native_res', 'processed');

% Find all subject directories
subject_dirs = dir(fullfile(processed_dir, 'sub-*'));
subject_dirs = subject_dirs([subject_dirs.isdir]);

fprintf('Segmenting MTsat for %d subject(s)\n', length(subject_dirs));
fprintf('========================================\n\n');

spm('defaults', 'FMRI');

for s = 1:length(subject_dirs)
    subject_name = subject_dirs(s).name;
    mpm_dir = fullfile(processed_dir, subject_name, 'mpm');

    if ~exist(mpm_dir, 'dir')
        fprintf('%s: No mpm/ directory, skipping\n', subject_name);
        continue;
    end

    fprintf('=== %s ===\n', subject_name);

    % Find MTsat file
    mtsat_pattern = fullfile(mpm_dir, [subject_name, '_MTsat.nii*']);
    mtsat_files = dir(mtsat_pattern);

    if isempty(mtsat_files)
        fprintf('  No MTsat file found, skipping\n\n');
        continue;
    end

    mtsat_file = fullfile(mtsat_files(1).folder, mtsat_files(1).name);

    % SPM12 requires uncompressed NIfTI
    if endsWith(mtsat_file, '.gz')
        mtsat_nii = mtsat_file(1:end-3);  % Remove .gz
        if ~exist(mtsat_nii, 'file')
            fprintf('  Uncompressing MTsat...\n');
            gunzip(mtsat_file, mpm_dir);
        end
    else
        mtsat_nii = mtsat_file;
    end

    fprintf('  Input: %s\n', mtsat_files(1).name);

    % Configure SPM segmentation batch
    matlabbatch = {};

    % Channel configuration: NO bias correction (biasreg = 10 = max value)
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {[mtsat_nii, ',1']};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 10;  % Suppress bias correction
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];

    % Tissue classes (6 classes, same as MPRAGE)
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
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
    matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
    matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                                  NaN NaN NaN];

    fprintf('  Running segmentation...\n');
    spm_jobman('run', matlabbatch);
    fprintf('  Done.\n\n');
end

fprintf('========================================\n');
fprintf('All segmentations complete.\n');
