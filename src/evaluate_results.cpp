#include "pudq_lib.hpp"
#include "graph.hpp"
#include "optimization.hpp"

#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <sstream>

std::vector<pudqlib::PUDQ> loadVerticesFromFile(const std::string& filename) {
    std::vector<pudqlib::PUDQ> vertices;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        pudqlib::PUDQ q;
        if (iss >> q(0) >> q(1) >> q(2) >> q(3)) {
            vertices.push_back(q);
        }
    }
    return vertices;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: ./evaluate_results <num_robots> <parent_export_dir>" << std::endl;
        return 1;
    }

    int num_robots = std::stoi(argv[1]);
    std::string parent_dir = argv[2];
    
    namespace fs = std::filesystem;
    std::string dir_without_cartan = (fs::path(parent_dir) / "without_cartan_sync").string();
    std::string dir_with_cartan    = (fs::path(parent_dir) / "with_cartan_sync").string();

    try {
        // =========================================================
        // CARTAN-SYNC 
        // =========================================================
        std::cout << "\n=========================================" << std::endl;
        std::cout << " PART 1: Evaluating Cartan-Sync Results" << std::endl;
        std::cout << "=========================================" << std::endl;

        if (fs::exists(dir_with_cartan)) {
            std::cout << "Loading Cartan graph from: " << dir_with_cartan << std::endl;
            pudqlib::Graph G_cartan = graph::load_graph_from_matlab_export(dir_with_cartan);

            const pudqlib::Result result_cartan = pudqlib::rgn_gradhess_J(G_cartan);
            const double& F_cartan = result_cartan.F_X;
            const Eigen::VectorXd& rgrad_cartan = result_cartan.rgrad_X;
            const double grad_norm_cartan = rgrad_cartan.norm();

            std::cout << "\n[Cartan-Sync]" << std::endl;
            std::printf("Final Cost      = %.6g\n", F_cartan);
            std::printf("Final Grad Norm = %.6g\n", grad_norm_cartan);
            
            graph::reorient_graph_to_identity(G_cartan);

            graph::calculate_ATE_Euclidean(G_cartan);
            graph::calculate_RPE_Euclidean(G_cartan);
            graph::calculate_ATE_PUDQ(G_cartan);
            graph::calculate_RPE_PUDQ(G_cartan);

        } else {
            std::cout << "WARNING: 'with_cartan_sync' folder not found. Skipping Cartan evaluation." << std::endl;
        }

        // =========================================================
        // DISTRIBUTED PGO 
        // =========================================================
        std::cout << "\n=========================================" << std::endl;
        std::cout << " PART 2: Evaluating Distributed PGO Results" << std::endl;
        std::cout << "=========================================" << std::endl;

        std::cout << "Loading base graph structure from: " << dir_without_cartan << std::endl;
        pudqlib::Graph global_graph = graph::load_graph_from_matlab_export(dir_without_cartan);
        
        std::cout << "Mapping robot ownership..." << std::endl;
        std::vector<pudqlib::MultiGraph> multi_graphs = graph::distributebiggraph_withinverseedge(global_graph, num_robots);

        std::string results_base_dir = "cpp_export_ros"; 
        std::string home_dir = std::getenv("HOME");
        std::string ros_default_dir = home_dir + "/.ros/cpp_export_ros";

        if (!fs::exists(results_base_dir) && fs::exists(ros_default_dir)) {
            std::cout << "Results not found in current directory. Found in ~/.ros/!" << std::endl;
            results_base_dir = ros_default_dir;
        }
        
        std::cout << "Merging distributed results from: '" << results_base_dir << "'..." << std::endl;
        
        bool missing_robot_data = false;
        for (int r = 0; r < num_robots; ++r) {
            fs::path robot_folder = fs::path(results_base_dir) / ("robot_" + std::to_string(r));
            std::string file_path = (robot_folder / "vertices_pudq.txt").string();
            
            if (!std::filesystem::exists(file_path)) {
                std::cerr << "WARNING: Missing result file for Robot " << r << " (" << file_path << ")" << std::endl;
                missing_robot_data = true;
                continue;
            }

            std::vector<pudqlib::PUDQ> local_optimized = loadVerticesFromFile(file_path);
            pudqlib::LocalGraph temp_lg = graph::build_local_subgraph(multi_graphs, r);
            const std::vector<int>& global_ids = temp_lg.vertices_interval;

            if (local_optimized.size() != global_ids.size()) {
                std::cerr << "ERROR: Size mismatch for Robot " << r << ". Expected " << global_ids.size() 
                          << ", got " << local_optimized.size() << std::endl;
                continue;
            }

            for (size_t i = 0; i < global_ids.size(); ++i) {
                int gid = global_ids[i];
                global_graph.vertices_pudq[gid] = local_optimized[i];
                global_graph.vertices[gid] = pudqlib::pudq_to_pose(local_optimized[i]);
            }
        }

        if (missing_robot_data) {
             std::cout << "Note: Some robot data was missing. Global graph is partially optimized." << std::endl;
        } else {
             std::cout << "Global graph merged successfully." << std::endl;
        }

        std::cout << "\n[DPGO]" << std::endl;
        const pudqlib::Result result_opt = pudqlib::rgn_gradhess_J(global_graph);
        const double& F_opt = result_opt.F_X;
        const Eigen::VectorXd& rgrad_opt = result_opt.rgrad_X;
        const double grad_norm = rgrad_opt.norm();
        
        std::printf("Final Cost      = %.6g\n", F_opt);
        std::printf("Final Grad Norm = %.6g\n", grad_norm);

        graph::reorient_graph_to_identity(global_graph); 

        graph::calculate_ATE_Euclidean(global_graph);
        graph::calculate_RPE_Euclidean(global_graph);
        graph::calculate_ATE_PUDQ(global_graph);
        graph::calculate_RPE_PUDQ(global_graph);

        std::cout << "--------------------------" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}