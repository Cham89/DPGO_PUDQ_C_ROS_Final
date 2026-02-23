clear, clc, close all;


% --- Setup Paths ---
run('/home/hsuanpin/DPGO_PUDQ_Matlab/DPGO_PUDQ_Matlab/Cartan-Sync/MATLAB/setup.m');
pudq_pgo_lib_path      = [pwd '/Pudq_lib'];
addpath(pudq_pgo_lib_path);

riem_opt_path          = [pwd '/riem_opt_lib'];
addpath(riem_opt_path);

se2_lib_path           = [pwd '/se2_lib'];
addpath(se2_lib_path);

Manopt_opts            = {};
SE_Sync_opts.init      = 'chordal';
SE_Sync_opts.Precon    = true;

% --- Parameters ---
num_robots = 5;
gt_g2o_file = '/home/hsuanpin/DPGO_PUDQ_Matlab/DPGO_PUDQ_Matlab/data/M3500_ground_truth.g2o';
odom_g2o_file = '/home/hsuanpin/DPGO_PUDQ_Matlab/DPGO_PUDQ_Matlab/data/M3500_5.g2o';
export_directory =  'matlab_export_c';

fprintf('Loading g2o files...\n');
GT_graph = load_g2o_file_pudq_gt (gt_g2o_file);
odom_graph = load_g2o_file_pudq (odom_g2o_file);
graph = combine_gt_odom_g2o(odom_graph, GT_graph);

% --- Chordal Initialization ---
fprintf('Running Chordal Initialization...\n');
G = graph;
num_vertices = length(graph.vertices);
[Rchordal, tchordal] = chordal_initialization(G);
for i=1:num_vertices
    R_i_index = 2*(i-1)+1;
    R_i = Rchordal(:,R_i_index:R_i_index+1);
    t_i = tchordal(:,i);
    theta_i = theta_to_R(R_i);
    pose_i = [t_i; theta_i];
    pudq_i = pose_to_pudq(pose_i);
    G.vertices{i} = pose_i;
    G.vertices_pudq{i} = pudq_i;
end
fprintf('Chordal Initialization complete.\n');

%%%%%%%%%% Cartan-Sync %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cartan_problem = SE_Sync_Cartan(G, Manopt_opts, SE_Sync_opts);
[SDPval_cartan, Yopt_cartan, Xhat_cartan, Fxhat_cartan, cartan_info] = run(cartan_problem);

%Convert SE sync data to graph
G_Cartan = se_sync_to_graph(G, Xhat_cartan);

%Get the final SE-Sync cost
F_Cartan = F_G_pudq(G_Cartan);

%Get SE-Sync time to convergence
cartan_time = sum([cartan_info.mat_construct_times, cartan_info.init_time, cartan_info.optimization_times, cartan_info.min_eig_vals, cartan_info.min_eig_times]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

export_dual_graphs(export_directory, G, G_Cartan);
fprintf('MATLAB processing finished. Data is ready for C++.\n');

G_Cartan_reorient         = reorient_graph(G_Cartan);
[cartan_rgrad, ~, cartan_F, ~] = rgn_gradhess_J(G_Cartan_reorient);
cartan_se2_cost = F_G_SE2(G_Cartan_reorient);
cartan_rpe_pudq = RPE_G_pudq(G_Cartan_reorient);
[cartan_rpe_eucl, cartan_rpe_linear, cartan_rpe_angular] = RPE_G_eucl(G_Cartan_reorient);
cartan_ate_pudq = ATE_G_pudq(G_Cartan_reorient);
[cartan_ate_eucl, cartan_ate_linear, cartan_ate_angular] = ATE_G_eucl(G_Cartan_reorient);


fprintf('Cartin-sync Results:\n');
fprintf('  SE2 Cost: %.6f\n', cartan_se2_cost);
fprintf('  PUDQ Cost: %.6f\n', cartan_F);
fprintf('  Grad: %.6f\n', norm(cartan_rgrad));
fprintf('  RPE PUDQ: %.6f\n', cartan_rpe_pudq);
fprintf('  RPE Euclidean: %.6f (Linear: %.6f, Angular: %.6f)\n', cartan_rpe_eucl, cartan_rpe_linear, cartan_rpe_angular);
fprintf('  ATE PUDQ: %.6f\n', cartan_ate_pudq);
fprintf('  ATE Euclidean: %.6f (Linear: %.6f, Angular: %.6f)\n', cartan_ate_eucl, cartan_ate_linear, cartan_ate_angular);

