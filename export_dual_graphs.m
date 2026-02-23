function export_dual_graphs(parent_directory, G_init, G_opt)
    
    dir_init = fullfile(parent_directory, 'without_cartan_sync');
    dir_opt  = fullfile(parent_directory, 'with_cartan_sync');

    if ~exist(dir_init, 'dir'), mkdir(dir_init); end
    if ~exist(dir_opt, 'dir'),  mkdir(dir_opt); end

    fprintf('------------------------------------------------\n');
    fprintf('Exporting "Without Cartan Sync" (Full Graph) to: %s\n', dir_init);
    export_graph_for_cpp(G_init, dir_init);

    fprintf('Exporting "With Cartan Sync" (Metrics Only) to:  %s\n', dir_opt);
    export_graph_for_cpp(G_opt, dir_opt);
    fprintf('------------------------------------------------\n');
end