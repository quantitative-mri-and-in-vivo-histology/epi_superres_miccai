%% create_hmri_maps.m
% Automatically loops through all subjects and creates hMRI maps
%
% This script processes all subjects in the single_pe_rpe dataset using
% SPM12's hMRI toolbox to generate Multi-Parameter Maps (MPM).

%% Setup paths
% Base data directory
base_dir = '../../data/single_pe_rpe/native_res';
source_dir = fullfile(base_dir, 'source', 'nifti_spm');
output_base_dir = fullfile(base_dir, 'processed');

% Create output directory if it doesn't exist
if ~exist(output_base_dir, 'dir')
    mkdir(output_base_dir);
end

%% Get list of subjects
subject_dirs = dir(fullfile(source_dir, 'sub-*'));
subject_dirs = subject_dirs([subject_dirs.isdir]);

fprintf('Found %d subjects to process\n', length(subject_dirs));

%% Initialize SPM
spm('defaults', 'FMRI');
spm_jobman('initcfg');

%% Loop through subjects
for subj_idx = 1:length(subject_dirs)

    subject_id = subject_dirs(subj_idx).name;
    subject_path = fullfile(source_dir, subject_id);

    fprintf('\n========================================\n');
    fprintf('Processing subject %d/%d: %s\n', subj_idx, length(subject_dirs), subject_id);
    fprintf('========================================\n');

    % Create subject-specific output directory
    output_dir = fullfile(output_base_dir, subject_id, 'mpm');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    %% Define scan folders (consistent across subjects)
    b1_folder = 'al_B1mapping_v2d_0004';
    b0_folder1 = 'gre_field_map3mmfilterM32channel_0002';
    b0_folder2 = 'gre_field_map3mmfilterM32channel_0003';
    mt_folder = 'nw_mtflash3d_v3hMT_0013';
    pd_folder = 'nw_mtflash3d_v3hPD_0010';
    t1_folder = 'nw_mtflash3d_v3hT1_0007';

    %% Check if all required folders exist
    required_folders = {b1_folder, b0_folder1, b0_folder2, mt_folder, pd_folder, t1_folder};
    folders_exist = true;
    for i = 1:length(required_folders)
        if ~exist(fullfile(subject_path, required_folders{i}), 'dir')
            fprintf('  WARNING: Missing folder %s\n', required_folders{i});
            folders_exist = false;
        end
    end

    if ~folders_exist
        fprintf('  SKIPPING subject %s due to missing folders\n', subject_id);
        continue;
    end

    %% Get B1 input files (all .nii files in B1 folder)
    b1_files = dir(fullfile(subject_path, b1_folder, '*.nii'));
    b1_input = cell(length(b1_files), 1);
    for i = 1:length(b1_files)
        b1_input{i} = [fullfile(subject_path, b1_folder, b1_files(i).name) ',1'];
    end
    fprintf('  Found %d B1 files\n', length(b1_input));

    %% Get B0 input files
    b0_files1 = dir(fullfile(subject_path, b0_folder1, '*.nii'));
    b0_files2 = dir(fullfile(subject_path, b0_folder2, '*.nii'));
    b0_input = cell(length(b0_files1) + length(b0_files2), 1);
    idx = 1;
    for i = 1:length(b0_files1)
        b0_input{idx} = [fullfile(subject_path, b0_folder1, b0_files1(i).name) ',1'];
        idx = idx + 1;
    end
    for i = 1:length(b0_files2)
        b0_input{idx} = [fullfile(subject_path, b0_folder2, b0_files2(i).name) ',1'];
        idx = idx + 1;
    end
    fprintf('  Found %d B0 files\n', length(b0_input));

    %% Get MT files
    mt_files = dir(fullfile(subject_path, mt_folder, '*.nii'));
    mt_input = cell(length(mt_files), 1);
    for i = 1:length(mt_files)
        mt_input{i} = [fullfile(subject_path, mt_folder, mt_files(i).name) ',1'];
    end
    fprintf('  Found %d MT files\n', length(mt_input));

    %% Get PD files
    pd_files = dir(fullfile(subject_path, pd_folder, '*.nii'));
    pd_input = cell(length(pd_files), 1);
    for i = 1:length(pd_files)
        pd_input{i} = [fullfile(subject_path, pd_folder, pd_files(i).name) ',1'];
    end
    fprintf('  Found %d PD files\n', length(pd_input));

    %% Get T1 files
    t1_files = dir(fullfile(subject_path, t1_folder, '*.nii'));
    t1_input = cell(length(t1_files), 1);
    for i = 1:length(t1_files)
        t1_input{i} = [fullfile(subject_path, t1_folder, t1_files(i).name) ',1'];
    end
    fprintf('  Found %d T1 files\n', length(t1_input));

    %% Build matlabbatch for this subject
    clear matlabbatch;

    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.output.outdir = {output_dir};
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.sensitivity.RF_us = '-';

    % B1 mapping
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI.b1input = b1_input;

    % B0 field maps
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI.b0input = b0_input;

    % B1 parameters
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI.b1parameters.b1metadata = 'yes';

    % Raw MPM data
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.raw_mpm.MT = mt_input;
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.raw_mpm.PD = pd_input;
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.raw_mpm.T1 = t1_input;

    % Popup option (set to false for batch processing)
    matlabbatch{1}.spm.tools.hmri.create_mpm.subj.popup = false;

    %% Run the batch
    fprintf('  Running hMRI create_mpm batch...\n');
    try
        spm_jobman('run', matlabbatch);
        fprintf('  SUCCESS: Subject %s processed successfully\n', subject_id);

        %% Clean up and reorganize files
        results_dir = fullfile(output_dir, 'Results');

        % Remove Supplementary folder
        supplementary_dir = fullfile(results_dir, 'Supplementary');
        if exist(supplementary_dir, 'dir')
            rmdir(supplementary_dir, 's');
        end

        % Remove _finished_ marker file
        finished_file = fullfile(results_dir, '_finished_');
        if exist(finished_file, 'file')
            delete(finished_file);
        end

        % Rename and move output files to parent directory (skip Results level)
        result_files = dir(fullfile(results_dir, '*.nii'));
        for f = 1:length(result_files)
            old_name = result_files(f).name;
            % Extract map type (everything after last underscore)
            parts = split(old_name, '_');
            map_type = strrep(parts{end}, '.nii', '');

            % New clean filename in parent directory
            new_name = sprintf('%s_%s.nii', subject_id, map_type);
            old_path = fullfile(results_dir, old_name);
            new_path = fullfile(output_dir, new_name);
            movefile(old_path, new_path);

            % Also move corresponding JSON file
            old_json = strrep(old_path, '.nii', '.json');
            new_json = strrep(new_path, '.nii', '.json');
            if exist(old_json, 'file')
                movefile(old_json, new_json);
            end
        end

        % Remove now-empty Results folder
        if exist(results_dir, 'dir')
            rmdir(results_dir);
        end

        fprintf('  Cleaned up and reorganized files\n');

    catch ME
        fprintf('  ERROR processing subject %s:\n', subject_id);
        fprintf('  %s\n', ME.message);
        continue;
    end

end

fprintf('\n========================================\n');
fprintf('Processing complete!\n');
fprintf('========================================\n');
