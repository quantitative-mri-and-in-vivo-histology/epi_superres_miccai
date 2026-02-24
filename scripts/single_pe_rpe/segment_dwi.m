% Segment DWI using SPM12 New Segment with eTPM — multi-channel
%
% Channel 1: mean b=0  — bias correction enabled  (biasreg = 0.001)
% Channel 2: DTI FA    — bias correction suppressed (biasreg = 10, max GUI value)
%
% Loops over all resolutions (native 1.6mm + downsampled 2.5mm),
% all subjects, and both processing modes (b0_rpe, full_rpe).
%
% Input files expected in each subject's dwi/ directory (produced by process_dwi.py):
%   {subject}_dir-{PE}_preprocessed_mean_b0.nii.gz    (b0_rpe mode)
%   {subject}_dir-{PE}_preprocessed_dti_fa.nii.gz     (b0_rpe mode)
%   {subject}_merged_preprocessed_mean_b0.nii.gz      (full_rpe mode)
%   {subject}_merged_preprocessed_dti_fa.nii.gz       (full_rpe mode)

script_dir = fileparts(mfilename('fullpath'));
root_dir   = fileparts(fileparts(script_dir));
tpm        = fullfile(root_dir, 'data', 'templates', 'eTPM.nii');

% Resolutions: {processed_base_dir, display_name}
resolutions = {
    fullfile(root_dir, 'data', 'single_pe_rpe', 'native_res',        'processed'), 'native 1.6mm';
    fullfile(root_dir, 'data', 'single_pe_rpe', 'downsampled_2p5mm', 'processed'), 'downsampled 2.5mm';
};

% Modes: {mode_name, file_pattern}
modes = {
    'b0_rpe',   '*_dir-*_preprocessed';
    'full_rpe', '*_merged_preprocessed';
};

spm('defaults', 'FMRI');

for r = 1:size(resolutions, 1)
    processed_dir = resolutions{r, 1};
    res_name      = resolutions{r, 2};

    if ~exist(processed_dir, 'dir')
        fprintf('Skipping %s (directory not found)\n', res_name);
        continue;
    end

    fprintf('\n=== %s ===\n', res_name);

    % Find all subject directories
    subject_dirs = dir(fullfile(processed_dir, 'sub-*'));
    subject_dirs = subject_dirs([subject_dirs.isdir]);

    for s = 1:length(subject_dirs)
        subject_name = subject_dirs(s).name;
        dwi_dir = fullfile(processed_dir, subject_name, 'dwi');

        if ~exist(dwi_dir, 'dir')
            fprintf('  %s: No dwi/ directory, skipping\n', subject_name);
            continue;
        end

        fprintf('  === %s ===\n', subject_name);

        for m = 1:size(modes, 1)
            mode_name    = modes{m, 1};
            file_pattern = modes{m, 2};

            fprintf('    Mode: %s\n', mode_name);

            % Find mean_b0 and dti_fa files matching the pattern
            mean_b0_pattern = fullfile(dwi_dir, [file_pattern '_mean_b0.nii*']);
            dti_fa_pattern  = fullfile(dwi_dir, [file_pattern '_dti_fa.nii*']);

            mean_b0_files = dir(mean_b0_pattern);
            dti_fa_files  = dir(dti_fa_pattern);

            if isempty(mean_b0_files)
                fprintf('      mean_b0 not found (pattern: %s), skipping\n', [file_pattern '_mean_b0.nii*']);
                continue;
            end
            if isempty(dti_fa_files)
                fprintf('      dti_fa not found (pattern: %s), skipping\n', [file_pattern '_dti_fa.nii*']);
                continue;
            end

            % For b0_rpe mode, there may be multiple PE directions - process all
            % For full_rpe mode, there should be only one merged file
            num_files = length(mean_b0_files);

            for f = 1:num_files
                mean_b0_file = fullfile(mean_b0_files(f).folder, mean_b0_files(f).name);
                dti_fa_file  = fullfile(dti_fa_files(f).folder, dti_fa_files(f).name);

                % Extract prefix for logging
                [~, prefix_b0, ~] = fileparts(mean_b0_files(f).name);
                prefix_b0 = strrep(prefix_b0, '_mean_b0', '');
                if endsWith(prefix_b0, '.nii')
                    prefix_b0 = prefix_b0(1:end-4);
                end

                if num_files > 1
                    fprintf('      Processing: %s\n', prefix_b0);
                end

                % Determine if file is compressed
                mean_b0_gz = endsWith(mean_b0_file, '.gz');
                dti_fa_gz  = endsWith(dti_fa_file, '.gz');

                % SPM12 requires uncompressed NIfTI
                if mean_b0_gz
                    mean_b0_nii = mean_b0_file(1:end-3);  % Remove .gz
                    if ~exist(mean_b0_nii, 'file')
                        fprintf('        Uncompressing mean_b0...\n');
                        gunzip(mean_b0_file, dwi_dir);
                    end
                else
                    mean_b0_nii = mean_b0_file;
                end

                if dti_fa_gz
                    dti_fa_nii = dti_fa_file(1:end-3);  % Remove .gz
                    if ~exist(dti_fa_nii, 'file')
                        fprintf('        Uncompressing dti_fa...\n');
                        gunzip(dti_fa_file, dwi_dir);
                    end
                else
                    dti_fa_nii = dti_fa_file;
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

                % Tissue classes (6 classes, same as multi_pe_rpe)
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

                fprintf('        Running segmentation...\n');
                spm_jobman('run', matlabbatch);
                fprintf('        Done.\n');
            end
        end
    end
end

fprintf('\nAll segmentations complete.\n');
