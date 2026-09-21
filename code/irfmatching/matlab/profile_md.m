function profile_md(t_grid, prior_csv)
% PRIOR_CSV (optional): a previous profile_md.csv; its scalar and factor solutions at grid points
% within 0.02 of t are added as extra starting values (a polish pass).
%PROFILE_MD Scalar refit / omitted-path exercise in m_d (household cognitive discounting).
%
% Set: h, xi_p, iota_p, xi_w, iota_w, kappa, m_d, m_f and the Frisch elasticity (global varphi)
% free; psi_u fixed at 0.5.  Population targets.  Weighted responses: rows of the standardized
% response matrix B_std (20 inflation horizons, then 20 output horizons; one column per
% one-s.d. factor score) divided by the per-row scale s.  Scalar target = B_std * c with |c| = 1;
% omitted-path target = B_std * c_perp, c_perp orthogonal to c; factor target = both columns.
% By construction Q_F = Q_S + Q_P.
%
% For each t: (1) hold m_d at 0.65 + t, refit the other eight to the scalar target -> theta_S(t),
% q_S(t); (2) evaluate the omitted-path loss at theta_S(t) with no refit -> q_P_at_S(t);
% (3) refit the eight to the factor target -> theta_F(t), q_F(t).  Also saves the baseline
% Jacobians (raw parameter units, weighted) for the local direction analysis, and the scalar-
% and omitted-path responses at each scalar refit.
if nargin < 1, t_grid = -0.10:0.025:0.10; end
if nargin < 2, prior_csv = ''; end
started = tic;
% Paths come from this file's own location, <bundle>/code/irfmatching/matlab.
bundle_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
data_dir = fullfile(bundle_root, 'data', 'irfmatching');
out_dir = fullfile(bundle_root, 'output', 'irfmatching');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
% The mp_modelcnfctls checkout supplies the model code; MP_MODELCNFCTLS overrides its location.
repo_root = getenv('MP_MODELCNFCTLS');
if isempty(repo_root), repo_root = fullfile(fileparts(bundle_root), 'mp_modelcnfctls'); end
saved = load(fullfile(data_dir, 'identification_demo.mat'));
result = saved.result;
addpath(fullfile(repo_root, '_auxiliary_functions'));
global T T_use n_shock shocks_match varphi %#ok<GVMIS>
T = result.options.TModel; T_use = max(result.options.LPi, result.options.LY); n_shock = 8; shocks_match = 1;
initialize_existing_rank(repo_root);
cfg = result.cfg; design = result.design; sd = result.scalar_direction;
n_macro = cfg.L_pi + cfg.L_y;
emp = struct('V_path_model', design.V_path_model, 'target_pc', zeros(n_macro * cfg.K_pc, 1), ...
    'Sigma_B_inv', eye(n_macro * cfg.K_pc), 'M_F', design.M_F);
param0 = [0.75 0.85 0.9 0.9 0.9 5.5 0.5 0.65 0.65 0.5]';
lb = [result.lower_bound(:); 0.05]; ub = [result.upper_bound(:); 3]; span = ub - lb;
names = {'h', 'xi_p', 'iota_p', 'xi_w', 'iota_w', 'kappa', 'psi_u', 'm_d', 'm_f', 'varphi'};
score_sd = sqrt(design.score_variances(:))';
i_md = 8; i_psi = 7;
    function B = B_std(p)
        varphi = p(10);
        raw = evaluate_h3_and_pc_moments(p(1:9)', emp, cfg, '/rank', sd);
        B = raw.B_pc .* score_sd;
    end
B0 = B_std(param0);
s = max(result.options.MacroScaleFloor, sqrt(mean(B0.^2, 2)));
c = sqrt(sd.denominator) * sd.a_h(:) ./ score_sd(:); c = c / norm(c);
c_perp = [-c(2); c(1)];
    function r = resid(p, kind)
        try
            E = (B_std(p) - B0) ./ s;
            switch kind
                case 'S', r = E * c;
                case 'P', r = E * c_perp;
                case 'F', r = E(:);
            end
        catch exception
            if strcmp(exception.identifier, 'evaluate_h3_and_pc_moments:InadmissibleParameters')
                if strcmp(kind, 'F'), r = 1e6 * ones(2 * n_macro, 1); else, r = 1e6 * ones(n_macro, 1); end
            else
                rethrow(exception);
            end
        end
    end
    function q = loss(p, kind), r = resid(p, kind); q = 0.5 * (r' * r); end

% --- baseline Jacobians over the nine free parameters (raw units, weighted) ---
free9 = setdiff(1:10, i_psi);
D_S = zeros(n_macro, 9); D_F = zeros(2 * n_macro, 9);
for j = 1:9
    k = free9(j); e = zeros(10, 1); e(k) = 2e-4 * max(abs(param0(k)), 0.1);
    Ep = (B_std(param0 + e) - B0) ./ s; Em = (B_std(param0 - e) - B0) ./ s;
    dE = (Ep - Em) / (2 * e(k));
    D_S(:, j) = dE * c; D_F(:, j) = dE(:);
end
save(fullfile(out_dir, 'profile_md_jacobians.mat'), 'D_S', 'D_F', 'c', 'c_perp', 's', 'B0', 'free9', 'names', 'param0');
fprintf('Jacobians done, %.0fs\n', toc(started));

% --- profiles ---
free = setdiff(1:10, [i_psi, i_md]);   % eight free in the refits
opts = optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', 1000, 'MaxFunctionEvaluations', 10000, ...
    'StepTolerance', 1e-9, 'FunctionTolerance', 1e-12, 'OptimalityTolerance', 1e-12, ...
    'FiniteDifferenceType', 'forward', 'FiniteDifferenceStepSize', 1e-5);
    function [best, q, flag] = refit(t, kind, starts)
        best = []; q = Inf; flag = NaN;
        for i = 1:numel(starts)
            p_start = starts{i}; p_start(i_md) = param0(i_md) + t; p_start(i_psi) = param0(i_psi);
            u0 = min(max((p_start(free) - lb(free)) ./ span(free), 0), 1);
            handle = @(u) resid(assemble(u, t), kind);
            [u, ss, ~, fl] = lsqnonlin(handle, u0, zeros(numel(free), 1), ones(numel(free), 1), opts);
            if 0.5 * ss < q, q = 0.5 * ss; best = assemble(u, t); flag = fl; end
        end
    end
    function p = assemble(u, t)
        p = param0; p(free) = lb(free) + span(free) .* u(:); p(i_md) = param0(i_md) + t;
    end
order = sort(t_grid, 'ComparisonMethod', 'abs');
theta_S = containers.Map('KeyType', 'double', 'ValueType', 'any');
theta_F = containers.Map('KeyType', 'double', 'ValueType', 'any');
fid = fopen(fullfile(out_dir, 'profile_md.csv'), 'w');
fprintf(fid, 't,q_S,q_P_at_S,q_F,flag_S,flag_F,%s,%s\n', strjoin(strcat('S_', names), ','), strjoin(strcat('F_', names), ','));
fid_irf = fopen(fullfile(out_dir, 'profile_md_irfs.csv'), 'w');
fprintf(fid_irf, 't,row,variable,horizon,scalar_baseline,scalar_refit,omitted_baseline,omitted_refit\n');
prior = [];
if ~isempty(prior_csv), prior = readtable(prior_csv); end
for t = order
    starts = {param0};
    if ~isempty(prior)
        rows = find(abs(prior.t - t) <= 0.02 + 1e-9);
        for rr = rows'
            starts{end + 1} = prior{rr, strcat('S_', names)}'; %#ok<AGROW>
            starts{end + 1} = prior{rr, strcat('F_', names)}'; %#ok<AGROW>
        end
    end
    if theta_S.Count > 0
        ks = cell2mat(keys(theta_S)); [~, i] = min(abs(ks - t)); near = ks(i);
        starts = [starts, {theta_S(near), theta_F(near)}];
    end
    [pS, qS, flS] = refit(t, 'S', starts);
    [pF, qF, flF] = refit(t, 'F', [starts, {pS}]);
    theta_S(t) = pS; theta_F(t) = pF;
    qP = loss(pS, 'P');
    fprintf(fid, '%g,%.8g,%.8g,%.8g,%d,%d,%s,%s\n', t, qS, qP, qF, flS, flF, ...
        strjoin(compose('%.8g', pS'), ','), strjoin(compose('%.8g', pF'), ','));
    B_refit = B_std(pS);
    sb = B0 * c; sr = B_refit * c; ob = B0 * c_perp; orf = B_refit * c_perp;
    for row = 1:n_macro
        if row <= cfg.L_pi, v = 'inflation'; hz = row - 1; else, v = 'output'; hz = row - cfg.L_pi - 1; end
        fprintf(fid_irf, '%g,%d,%s,%d,%.8g,%.8g,%.8g,%.8g\n', t, row, v, hz, sb(row), sr(row), ob(row), orf(row));
    end
    fprintf('t %+.3f | q_S %.3g | q_P_at_S %.3g | q_F %.3g | flags %d %d | %.0fs\n', t, qS, qP, qF, flS, flF, toc(started));
end
fclose(fid); fclose(fid_irf);
fprintf('FINISHED %.0fs\n', toc(started));
end

function initialize_existing_rank(repo_root)
path = repo_root; vintage = ''; model = '/rank'; load_hank = 1; %#ok<NASGU>
auxiliary_dir = fullfile(repo_root, '_auxiliary_functions');
run(fullfile(auxiliary_dir, 'get_calibration_general.m'));
run(fullfile(auxiliary_dir, 'get_baseline_jacobians.m'));
run(fullfile(auxiliary_dir, 'get_priors_general.m'));
end
