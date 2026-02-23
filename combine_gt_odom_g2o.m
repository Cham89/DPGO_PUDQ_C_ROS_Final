function graph = combine_gt_odom_g2o(G_odom, G_gt)
    size_edge = size(G_gt.edges);
    num_edges = size_edge(1);

    graph = G_odom;
    graph.vertices_true = G_gt.vertices_true;
    graph.vertices_pudq_true = G_gt.vertices_pudq_true;
    graph.delta_poses_true = G_gt.delta_poses;
    graph.delta_poses_pudq_true = G_gt.delta_poses_pudq;

    for ij = 1: num_edges
        z_ij_true = G_gt.delta_poses_pudq{ij};
        z_ij_noisy = graph.delta_poses_pudq{ij};
        exp_eta_ij = pudq_compose(pudq_inv(z_ij_true), z_ij_noisy);
        eta_ij = Log_1(exp_eta_ij);

        information_euc = graph.information{ij};
        information_pudq = info_eucl_to_pudq(information_euc, 2 * eta_ij(1));
        information_se2 = info_pudq_to_se2(information_pudq);
        graph.information_pudq{ij} = information_pudq;
        graph.information_se2{ij} = information_se2;
    end

    graph.Omega = Omega_sparse(graph);
end
