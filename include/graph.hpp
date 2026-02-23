#pragma once
#include "pudq_lib.hpp"

#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <utility>            
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace graph{

    pudqlib::Graph load_g2o_file_pudq(const std::string& g2o_file);
    pudqlib::Graph load_g2o_file_pudq_gt(const std::string& g2o_file);
    pudqlib::Graph combine_gt_odom_g2o(const pudqlib::Graph& G_odom, const pudqlib::Graph& G_gt);

    int find_local_vertex_index (const pudqlib::MultiGraph& rg, int g_index);
    int find_robot_for_vertex (const std::vector<pudqlib::MultiGraph>& mg, int num_robots, int g_vertex); 
    std::pair<int,bool> findOrAddLandmark(pudqlib::MultiGraph& rg, int r_j, int v_j_local);
    std::vector<pudqlib::MultiGraph> distributebiggraph_withinverseedge(const pudqlib::Graph& graph, int num_robots);
    pudqlib::LocalGraph build_local_subgraph(const std::vector<pudqlib::MultiGraph>& multi_graph, int r);
    pudqlib::Graph reconstructglobalgraph(const std::vector<pudqlib::Robot>& robots, const pudqlib::Graph& original_G);
    Eigen::MatrixXd readMatrixFromFile(const std::string& filename);
    pudqlib::Graph load_graph_from_matlab_export(const std::string& directory_path);

    template<typename T>
    void writeEigenVectorListToFile(const std::string& filename, const std::vector<T>& vec) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        file << std::fixed << std::setprecision(16);
        for (const auto& v : vec) {
            file << v.transpose() << "\n";
        }
    }

    void saveFullAnalysisData(const std::string& output_dir,const pudqlib::Graph& optimized_graph, const std::vector<pudqlib::Robot>& robots);
    
    double wrap_theta(double theta);
    double theta_diff(double theta_0, double theta_1);
    

    void reorient_graph_to_identity(pudqlib::Graph& G);
    void align_graph_to_ground_truth(pudqlib::Graph& G);

    void calculate_ATE_Euclidean(const pudqlib::Graph& G);
    void calculate_RPE_Euclidean(const pudqlib::Graph& G);
    void calculate_ATE_PUDQ(const pudqlib::Graph& G);
    void calculate_RPE_PUDQ(const pudqlib::Graph& G);




}