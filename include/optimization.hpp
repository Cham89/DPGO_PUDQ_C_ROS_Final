#pragma once
#include "pudq_lib.hpp"

#include <tuple> 
#include <vector>
#include <cmath>
#include <string>
#include <utility>            
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace opt{
    std::tuple<double, double> Optimization_RLM_inverse_edge(pudqlib::LocalGraph& local_graph, double lambda_in, double grad_tol);
    void shareupdateinfo(int share_robot_id, const pudqlib::LocalGraph& share_subgraph, std::vector<pudqlib::Robot>&  robotList);
    void check_all_robotgrad_inverse_edge(std::vector<pudqlib::Robot>& robotlist, double grad_tol);
}
