#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <utility>            
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>

namespace pudqlib{
    
    using PUDQ = Eigen::Matrix<double,4,1>;

    struct Graph {
        std::vector<Eigen::Vector3d>    vertices;
        std::vector<PUDQ>               vertices_pudq;
        std::vector<Eigen::Vector3d>    vertices_true;
        std::vector<PUDQ>               vertices_pudq_true;

        std::vector<std::pair<int,int>> edges;
        std::vector<Eigen::Vector3d>    delta_poses;
        std::vector<PUDQ>               delta_poses_pudq;
        std::vector<Eigen::Vector3d>    delta_poses_true;
        std::vector<PUDQ>               delta_poses_pudq_true;
        std::vector<Eigen::Vector2d>    t;
        std::vector<Eigen::Matrix2d>    R;

        std::vector<Eigen::Matrix3d>    information;
        std::vector<Eigen::Matrix3d>    information_pudq;
        std::vector<Eigen::Matrix3d>    information_se2;

        std::vector<double>             kappa;
        std::vector<double>             tau;

        Eigen::SparseMatrix<double>     Omega;
    };

    struct InterEdge{
        int i = -1; 
        int j = -1; 
        bool flag = false;
    };

    struct ForeignInfo{
        int robot = -1; 
        int local_index = -1;
        int global_index = -1; 
    };

    struct MultiGraph{
        std::vector<Eigen::Vector3d>    vertices_true;
        std::vector<PUDQ>               vertices_pudq_true;
        std::vector<Eigen::Vector3d>    vertices;
        std::vector<PUDQ>               vertices_pudq;
        std::vector<int>                vertices_interval; 

        std::vector<Eigen::Vector2i>    intra_edges;
        std::vector<Eigen::Vector3d>    intra_dp;
        std::vector<Eigen::Vector3d>    intra_dp_true;
        std::vector<PUDQ>               intra_dp_pudq;
        std::vector<PUDQ>               intra_dp_pudq_true;
        std::vector<Eigen::Vector2d>    intra_t;
        std::vector<Eigen::Matrix2d>    intra_R;
        std::vector<Eigen::Matrix3d>    intra_info;
        std::vector<Eigen::Matrix3d>    intra_info_pudq;
        std::vector<Eigen::Matrix3d>    intra_info_se2;
        std::vector<double>             intra_tau;
        std::vector<double>             intra_kappa;

        std::vector<InterEdge>          inter_edges;
        std::vector<Eigen::Vector3d>    inter_dp;
        std::vector<Eigen::Vector3d>    inter_dp_true;
        std::vector<PUDQ>               inter_dp_pudq;
        std::vector<PUDQ>               inter_dp_pudq_true;
        std::vector<Eigen::Vector2d>    inter_t;
        std::vector<Eigen::Matrix2d>    inter_R;
        std::vector<Eigen::Matrix3d>    inter_info;
        std::vector<Eigen::Matrix3d>    inter_info_pudq;
        std::vector<Eigen::Matrix3d>    inter_info_se2;
        std::vector<double>             inter_tau;
        std::vector<double>             inter_kappa;

        std::vector<Eigen::Vector3d>    lm_vertices;
        std::vector<PUDQ>               lm_vertices_pudq;
        std::vector<Eigen::Vector3d>    lm_vertices_true;
        std::vector<PUDQ>               lm_vertices_pudq_true;
        std::vector<ForeignInfo>        lm_foreign_info;

        Eigen::SparseMatrix<double>     Omega_ij;
    };

    struct LocalGraph{
        int robot_id = -1;
        bool anchor_first = false;

        std::vector<Eigen::Vector3d>    vertices_true;
        std::vector<pudqlib::PUDQ>      vertices_pudq_true;
        std::vector<Eigen::Vector3d>    vertices;
        std::vector<pudqlib::PUDQ>      vertices_pudq;
        std::vector<int>                vertices_interval; 

        std::vector<Eigen::Vector2i>    intra_edges;
        std::vector<Eigen::Vector3d>    intra_dp;
        std::vector<Eigen::Vector3d>    intra_dp_true;
        std::vector<pudqlib::PUDQ>      intra_dp_pudq;
        std::vector<pudqlib::PUDQ>      intra_dp_pudq_true;
        std::vector<Eigen::Vector2d>    intra_t;
        std::vector<Eigen::Matrix2d>    intra_R;
        std::vector<Eigen::Matrix3d>    intra_info;
        std::vector<Eigen::Matrix3d>    intra_info_pudq;
        std::vector<Eigen::Matrix3d>    intra_info_se2;
        std::vector<double>             intra_tau;
        std::vector<double>             intra_kappa;

        std::vector<InterEdge>          inter_edges;
        std::vector<Eigen::Vector3d>    inter_dp;
        std::vector<Eigen::Vector3d>    inter_dp_true;
        std::vector<pudqlib::PUDQ>      inter_dp_pudq;
        std::vector<pudqlib::PUDQ>      inter_dp_pudq_true;
        std::vector<Eigen::Vector2d>    inter_t;
        std::vector<Eigen::Matrix2d>    inter_R;
        std::vector<Eigen::Matrix3d>    inter_info;
        std::vector<Eigen::Matrix3d>    inter_info_pudq;
        std::vector<Eigen::Matrix3d>    inter_info_se2;
        std::vector<double>             inter_tau;
        std::vector<double>             inter_kappa;

        std::vector<Eigen::Vector3d>    lm_vertices;
        std::vector<pudqlib::PUDQ>      lm_vertices_pudq;
        std::vector<Eigen::Vector3d>    lm_vertices_true;
        std::vector<pudqlib::PUDQ>      lm_vertices_pudq_true;
        std::vector<ForeignInfo>        lm_foreign_info;

        Eigen::SparseMatrix<double>     Omega_ij;
    };

    struct Result{
        Eigen::VectorXd                 rgrad_X;
        Eigen::SparseMatrix<double>     rgnhess_X;
        double                          F_X;
        Eigen::SparseMatrix<double>     P_X;
    };

    struct Robot{
        int id = -1;
        LocalGraph local_graph;
        std::vector<double> rgn_cost;
        std::vector<double> rgn_gradnorm;
        bool converge = false;
    };

    Eigen::Matrix4d Q_L(const PUDQ& x);
    double sinc1(double x); 
    double cosc(double x);
    double get_phi_atan2(double sin_phi, double cos_phi);
    Eigen::Vector3d randomdelta(double straight_prob);
    double f_1(double x);
    Eigen::Matrix4d P_x(const PUDQ& x);

    PUDQ pudq_normalize(const PUDQ& q);
    PUDQ pudq_mul(const PUDQ& a, const PUDQ& b);
    PUDQ pudq_inv(const PUDQ& q); 
    PUDQ pudq_compose(const PUDQ& a, const PUDQ& b); 
    Eigen::Matrix3d generate_pudq_cov(double sigma, int df); //這個極度不確定 但只有 generate gridworld 時會用到
    
    Eigen::Vector3d Log_1(const PUDQ& c);    
    PUDQ Exp_1(const Eigen::Vector3d& x_t);
    PUDQ Exp_x(const PUDQ& x_i, const Eigen::Vector4d& y_t); 
    Eigen::VectorXd Exp_X_N(const Eigen::VectorXd& X, const Eigen::VectorXd& Y_T);

    Eigen::Matrix2d R_from_theta(double th); 
    double theta_from_R(const Eigen::Matrix2d& R);
    
    PUDQ pose_to_pudq(const Eigen::Vector3d& pose); 
    Eigen::Vector3d pudq_to_pose(const PUDQ& q);  

    Eigen::Matrix3d info_eucl_to_pudq(const Eigen::Matrix3d& info_eucl, double theta_error); 
    Eigen::Matrix3d info_eucl_to_se2(const Eigen::Matrix3d& info_eucl, double theta_error); 
    Eigen::Matrix3d info_pudq_to_se2(const Eigen::Matrix3d& info_pudq); 
    Eigen::Matrix3d info_pudq_to_eucl(const Eigen::Matrix3d& info_pudq, double theta_error); 

    Eigen::SparseMatrix<double> Omega_sparse(const Graph& biggraph); 
    Eigen::SparseMatrix<double> Omega_sparse_Multi(const MultiGraph& robotgraph);

    Eigen::SparseMatrix<double> P_X_N_sparse(const std::vector<PUDQ>& X);
    Result rgn_gradhess_multi_J_forinverse(const LocalGraph& local_graph);
    Result rgn_gradhess_J(const Graph& G);

    Eigen::VectorXd G_get_X(const LocalGraph& G);
    LocalGraph G_set_X(LocalGraph& G, const Eigen::VectorXd& X_new);

}