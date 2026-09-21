function norefit_md(t_grid)
%NOREFIT_MD Loss when m_d is moved and nothing else is refit, under both target sets.
% Same construction as profile_md (Frisch free set irrelevant here since nothing is refit).
if nargin < 1, t_grid = [-0.04 -0.03 -0.02 -0.01 -0.005 0.005 0.01 0.02 0.03 0.04]; end
% Paths come from this file's own location, <bundle>/code/irfmatching/matlab.
bundle_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
data_dir = fullfile(bundle_root, 'data', 'irfmatching');
out_dir = fullfile(bundle_root, 'output', 'irfmatching');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
% The mp_modelcnfctls checkout supplies the model code; MP_MODELCNFCTLS overrides its location.
repo_root = getenv('MP_MODELCNFCTLS');
if isempty(repo_root), repo_root = fullfile(fileparts(bundle_root), 'mp_modelcnfctls'); end
saved = load(fullfile(data_dir, 'identification_demo.mat')); result = saved.result;
addpath(fullfile(repo_root, '_auxiliary_functions'));
global T T_use n_shock shocks_match varphi %#ok<GVMIS>
T = result.options.TModel; T_use = max(result.options.LPi, result.options.LY); n_shock = 8; shocks_match = 1;
initialize_existing_rank(repo_root);
cfg = result.cfg; design = result.design; sd = result.scalar_direction;
n_macro = cfg.L_pi + cfg.L_y;
emp = struct('V_path_model', design.V_path_model, 'target_pc', zeros(n_macro * cfg.K_pc, 1), ...
    'Sigma_B_inv', eye(n_macro * cfg.K_pc), 'M_F', design.M_F);
param0 = [0.75 0.85 0.9 0.9 0.9 5.5 0.5 0.65 0.65 0.5]';
score_sd = sqrt(design.score_variances(:))';
    function B = B_std(p)
        varphi = p(10);
        raw = evaluate_h3_and_pc_moments(p(1:9)', emp, cfg, '/rank', sd);
        B = raw.B_pc .* score_sd;
    end
B0 = B_std(param0);
s = max(result.options.MacroScaleFloor, sqrt(mean(B0.^2, 2)));
c = sqrt(sd.denominator) * sd.a_h(:) ./ score_sd(:); c = c / norm(c);
fid = fopen(fullfile(out_dir, 'norefit_md.csv'), 'w'); fprintf(fid, 't,q_S,q_F\n');
for t = t_grid
    p = param0; p(8) = param0(8) + t;
    E = (B_std(p) - B0) ./ s;
    qS = 0.5 * norm(E * c)^2; qF = 0.5 * norm(E(:))^2;
    fprintf(fid, '%g,%.8g,%.8g\n', t, qS, qF); fprintf('t %+.3f q_S %.4g q_F %.4g\n', t, qS, qF);
end
fclose(fid);
end

function initialize_existing_rank(repo_root)
path = repo_root; vintage = ''; model = '/rank'; load_hank = 1; %#ok<NASGU>
auxiliary_dir = fullfile(repo_root, '_auxiliary_functions');
run(fullfile(auxiliary_dir, 'get_calibration_general.m'));
run(fullfile(auxiliary_dir, 'get_baseline_jacobians.m'));
run(fullfile(auxiliary_dir, 'get_priors_general.m'));
end
