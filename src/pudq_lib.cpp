#include "pudq_lib.hpp"
#include <Eigen/Sparse>  
#include <Eigen/Core>  
#include <vector>
#include <cmath>

namespace pudqlib{

    Eigen::Matrix4d Q_L(const PUDQ& x){
        Eigen::Matrix4d Q;
        Q << x(0), -x(1), 0.0, 0.0, 
             x(1),  x(0), 0.0, 0.0,
             x(2),  x(3), x(0), -x(1),
             x(3), -x(2), x(1),  x(0);
        return Q;
    }

    double sinc1(double x){
        if (x == 0.0){
            return 1.0;
        } else{
            return std::sin(x)/x;
        }
    }

    double cosc(double x){
        if (x == 0.0){
            return 0.0;
        } else{
            return (1.0 - std::cos(x)) / x;
        }
    }

    double get_phi_atan2(double sin_phi, double cos_phi){
        double phi = std::atan2(sin_phi, cos_phi);
        if (phi <= -M_PI / 2.0){
            phi = phi + M_PI;
        } else if (phi > M_PI / 2.0){
            phi = phi - M_PI;
        }
        return phi;
    }
    
/*     Eigen::Vector3d randomdelta(double straight_prob){
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> U01(0.0, 1.0);

        const double random_prob = U01(rng);
        const double turn_prob = straight_prob + (1.0 - straight_prob) / 2.0;

        if (random_prob <= straight_prob) {
            return Eigen::Vector3d(1.0, 0.0, 0.0);
        } else if (random_prob <= turn_prob) {
            return Eigen::Vector3d(0.0, 1.0, std::numbers::pi/2.0);
        } else {
            return Eigen::Vector3d(0.0, -1.0, -(std::numbers::pi/2.0));
        }
    } */

    double f_1(double x){
        if (x == 0.0){
            return 0.0;
        } else{
            const double s = std::sin(x);
            const double c = std::cos(x);
            return (1.0 / (s * s)) * (s - x * c);
        }
    }

    Eigen::Matrix4d P_x(const PUDQ& x){
        const Eigen::Vector2d x_r = x.head<2>();
        Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
        P.topLeftCorner<2,2>() = Eigen::Matrix2d::Identity() - x_r * x_r.transpose();
        P.bottomRightCorner<2,2>() = Eigen::Matrix2d::Identity();
        return P;
    }

    PUDQ pudq_normalize(const PUDQ& q){
        PUDQ p = q;
        p.head<2>() = q.head<2>() / q.head<2>().norm();
        return p;
    }

    PUDQ pudq_mul(const PUDQ& a, const PUDQ& b){
        PUDQ p = Q_L(a) * b;
        return p;
    }

    PUDQ pudq_inv(const PUDQ& q){
        PUDQ q_inv;
        q_inv(0) =  q(0);
        q_inv(1) = -q(1);
        q_inv(2) = -q(2);
        q_inv(3) = -q(3);
        return q_inv;
    }

    PUDQ pudq_compose(const PUDQ& a, const PUDQ& b){
        PUDQ p = pudq_normalize(pudq_mul(a, b));
        return p;
    }

    //這個還沒確認 極度不確定 但只有 generate gridworld 時會用到
/*     Eigen::Matrix3d generate_pudq_cov(double sigma_pudq, int df) {
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_real_distribution<double> U(std::numeric_limits<double>::min(), 1.0);

        Eigen::Vector3d u(U(rng), U(rng), U(rng));
        Eigen::Matrix3d Sigma_w = Eigen::Matrix3d::Ones();
        Sigma_w.diagonal().array() += u.array();  

        const Eigen::Matrix3d S = sigma_pudq * Sigma_w;

        if (df < 3) throw std::runtime_error("Wishart df must be >= dimension (3)");

        Eigen::LLT<Eigen::Matrix3d> cholS(S);
        if (cholS.info() != Eigen::Success) throw std::runtime_error("S not PD in generate_pudq_cov");
        const Eigen::Matrix3d L = cholS.matrixL();

        std::normal_distribution<double> N01(0.0, 1.0);
        auto chi2 = [&](int nu) {
            // chi2_nu ~ Gamma(k=nu/2, theta=2)
            std::gamma_distribution<double> G(0.5 * nu, 2.0);
            return G(rng);
        };

        Eigen::Matrix3d T = Eigen::Matrix3d::Zero();
        T(0,0) = std::sqrt(chi2(df - 0));
        T(1,0) = N01(rng);
        T(1,1) = std::sqrt(chi2(df - 1));
        T(2,0) = N01(rng);
        T(2,1) = N01(rng);
        T(2,2) = std::sqrt(chi2(df - 2));

        const Eigen::Matrix3d C = L * T;
        const Eigen::Matrix3d W = C * C.transpose();

        return W / static_cast<double>(df);
    } */

    Eigen::Vector3d Log_1(const PUDQ& r){
        double phi = get_phi_atan2(r(1), r(0));
        double gamma = sinc1(phi);
        Eigen::Vector3d x_t;
        x_t << r(1) / gamma,
               r(2) / gamma,
               r(3) / gamma;
        return x_t;
    }

    PUDQ Exp_1(const Eigen::Vector3d& x_t){
        const double phi = x_t(0);
        const double gamma = sinc1(phi);
        PUDQ q;
        q << std::cos(phi),            
            gamma * phi,             
            gamma * x_t(1),           
            gamma * x_t(2);          
        return pudq_normalize(q);
    }

    PUDQ Exp_x(const PUDQ& x, const Eigen::Vector4d& y_t_raw) {

        Eigen::Vector4d y_t = P_x(x) * y_t_raw;
        Eigen::Matrix2d Q_Lx;
        Q_Lx << x(0), -x(1),
                x(1),  x(0);
        
        double phi_r = -x(1) * y_t(0) + x(0) * y_t(1);
        
        Eigen::Vector2d trig_vec;
        trig_vec << std::cos(phi_r), std::sin(phi_r);
        Eigen::Vector2d exp_yr = Q_Lx * trig_vec;
        
        PUDQ exp_x;
        exp_x << exp_yr, x(2) + y_t(2), x(3) + y_t(3);
        
        return exp_x;
    }

    Eigen::VectorXd Exp_X_N(const Eigen::VectorXd& X, const Eigen::VectorXd& Y_T){
        const int num_vertices = static_cast<int>(X.size()) / 4;
        Eigen::VectorXd exp_x_n = Eigen::VectorXd::Zero(num_vertices * 4);

        for (int i = 0; i < num_vertices; ++i){
            const int idx = 4 * i;
            const PUDQ x_i  = X.segment<4>(idx);   
            const Eigen::Vector4d y_t  = Y_T.segment<4>(idx);  
            exp_x_n.segment<4>(idx) = Exp_x(x_i, y_t); 
        }
        return exp_x_n;
    }
    
    Eigen::Matrix2d R_from_theta(double th){
        const double c = std::cos(th), s = std::sin(th);
        Eigen::Matrix2d R;
        R << c, -s,
             s,  c;
        return R;
    }

    double theta_from_R(const Eigen::Matrix2d& R){
        return std::atan2(R(1,0), R(0,0));
    }

    PUDQ pose_to_pudq(const Eigen::Vector3d& pose){
        Eigen::Vector2d t = pose.head<2>();
        double theta = pose(2);
        PUDQ pudq;
        double c = std::cos(theta/2.0);
        double s = std::sin(theta/2.0);
        pudq(0) = c;
        pudq(1) = s;
        Eigen::Matrix2d R_half;
        R_half << c, s, -s, c;
        pudq.tail<2>() = 0.5 * R_half * t;
        return pudq;
    }

    Eigen::Vector3d pudq_to_pose(const PUDQ& q){
        const double theta = 2.0 * get_phi_atan2(q(1), q(0));
        Eigen::Matrix2d R_phi;
        R_phi << q(0),  q(1),
                -q(1),  q(0);
        const Eigen::Vector2d t = 2.0 * R_phi.transpose() * q.tail<2>();
        return Eigen::Vector3d(t.x(), t.y(), theta);
    } 

    Eigen::Matrix3d info_eucl_to_pudq(const Eigen::Matrix3d& info_eucl, double theta_error){
        const double alpha = theta_error / 2.0;
        const double beta = std::cos(alpha) / sinc1(alpha);

        Eigen::Matrix3d M;
        M << 1.0, 0.0, 0.0,
             0.0, beta, alpha,
             0.0, -alpha, beta;

        Eigen::Matrix3d B;
        B << 0.0, 0.0, 1.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0;
    
        Eigen::Matrix3d info_pudq = 4.0 * B * ((M.transpose().inverse() * info_eucl) * M.inverse()) * B.transpose();
        info_pudq = info_pudq.triangularView<Eigen::Lower>(); 
        info_pudq += info_pudq.triangularView<Eigen::StrictlyLower>().transpose();
        return info_pudq;
    }

    Eigen::Matrix3d info_eucl_to_se2(const Eigen::Matrix3d& info_eucl, double theta_error){
        const double s = sinc1(theta_error);
        const double c = cosc(theta_error);

        Eigen::Matrix3d M;
        M << s,  -c,   0.0,
             c,   s,   0.0,
             0.0, 0.0, 1.0;

        Eigen::Matrix3d out = M.transpose() * info_eucl * M;
        out = 0.5 * (out + out.transpose());     
        return out;
    }

    Eigen::Matrix3d info_pudq_to_se2(const Eigen::Matrix3d& info_pudq){
        Eigen::Matrix3d B;
        B << 0.0, 0.0, 1.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0;

        Eigen::Matrix3d info_se2 = 0.25 * B.transpose() * info_pudq * B;
        return info_se2;
    }

    Eigen::Matrix3d info_pudq_to_eucl(const Eigen::Matrix3d& info_pudq, double theta_error){
        const double alpha = theta_error / 2.0;
        const double beta  = std::cos(alpha) / sinc1(alpha);

        Eigen::Matrix3d M;
        M << 1.0,  0.0,   0.0,
             0.0,  beta,  alpha,
             0.0, -alpha, beta;

        Eigen::Matrix3d B;
        B << 0.0, 0.0, 1.0,
             1.0, 0.0, 0.0,
             0.0, 1.0, 0.0;

        Eigen::Matrix3d info_eucl = 0.25 * M.transpose() * B.transpose() * info_pudq * B * M;
        return info_eucl;
    }
    
    Eigen::SparseMatrix<double> Omega_sparse(const Graph& biggraph){
        int M = biggraph.information_pudq.size();
        int num_triplets = 9 * M;

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(num_triplets);

        for (int ij = 0; ij < M; ++ij){
            const Eigen::Matrix3d& info_pudq = biggraph.information_pudq[ij];

            for (int i = 0; i < 3; ++i){
                for (int j = 0; j < 3; ++j){
                    int row_idx = 3 * ij + i;
                    int col_idx = 3 * ij + j;
                    double value = info_pudq(i, j);

                    triplets.emplace_back(row_idx, col_idx, value);
                }
            }
        }

        Eigen::SparseMatrix<double> Omega(3 * M, 3 * M);
        Omega.setFromTriplets(triplets.begin(), triplets.end());

        return Omega;
    }

    Eigen::SparseMatrix<double> Omega_sparse_Multi(const MultiGraph& robotgraph){
        int M_intra = robotgraph.intra_info_pudq.size();
        int M_inter = robotgraph.inter_info_pudq.size();
        int M = M_intra + M_inter;
        int num_triplets = 9 * M;

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(num_triplets);

        std::vector<Eigen::Matrix3d> all_info_pudq;
        all_info_pudq.reserve(M);
        all_info_pudq.insert(all_info_pudq.end(), robotgraph.intra_info_pudq.begin(), robotgraph.intra_info_pudq.end());
        all_info_pudq.insert(all_info_pudq.end(), robotgraph.inter_info_pudq.begin(), robotgraph.inter_info_pudq.end());

        for (int ij = 0; ij < M; ++ij){
            double factor = (ij < M_intra) ? 1.0 : 0.5;
            const Eigen::Matrix3d& info_pudq = factor *all_info_pudq[ij];

            for (int i = 0; i < 3; ++i){
                for (int j = 0; j < 3; ++j){
                    int row_idx = 3 * ij + i;
                    int col_idx = 3 * ij + j;
                    double value = info_pudq(i, j);

                    triplets.emplace_back(row_idx, col_idx, value);
                }
            }
        }

        Eigen::SparseMatrix<double> Omega_ij(3 * M, 3 * M);
        Omega_ij.setFromTriplets(triplets.begin(), triplets.end());

        return Omega_ij;
    }

    Eigen::SparseMatrix<double> P_X_N_sparse(const std::vector<PUDQ>& X){
        const int N = static_cast<int>(X.size());
        const int dim = 4 * N;

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(16 * N);
        
        for (int v = 0; v < N; ++v){
            const Eigen::Vector2d vv = X[v].head<2>();
            const Eigen::Matrix2d I2 = Eigen::Matrix2d::Identity();
            const int base = 4 * v;
            
            Eigen::Matrix4d P_x = Eigen::Matrix4d::Zero();
            P_x.block<2,2>(0,0) = I2 - vv * vv.transpose();
            P_x.block<2,2>(2,2) = I2;

            for (int i = 0; i < 4; ++i){
                for (int j = 0; j < 4; ++j){
                    trips.emplace_back(base + i, base + j, P_x(i, j));
                }
            }
        }

        Eigen::SparseMatrix<double> P(dim, dim);
        P.setFromTriplets(trips.begin(), trips.end());
        return P;
    }

    Result rgn_gradhess_multi_J_forinverse(const LocalGraph& local_graph){
        const int N = static_cast<int>(local_graph.vertices.size());
        const int num_intra_edges = static_cast<int>(local_graph.intra_edges.size());
        const int num_inter_edges = static_cast<int>(local_graph.inter_edges.size());

        const int num_J_triplets = 24 * num_intra_edges + 12 * num_inter_edges;
        std::vector<Eigen::Triplet<double>> J_triplets;
        J_triplets.reserve(num_J_triplets);

        const int num_E_triplets = 3 * (num_intra_edges + num_inter_edges);
        std::vector<Eigen::Triplet<double>> E_triplets;
        E_triplets.reserve(num_E_triplets);

        // Intra edges
        for (int ij = 0; ij < num_intra_edges; ++ij){ 
            int edge_i = local_graph.intra_edges[ij](0);
            int edge_j = local_graph.intra_edges[ij](1);
            PUDQ x_i = local_graph.vertices_pudq[edge_i];
            PUDQ x_j = local_graph.vertices_pudq[edge_j];
            PUDQ z_ij = local_graph.intra_dp_pudq[ij];

            PUDQ r_ij = pudq_compose(pudq_inv(z_ij), pudq_mul(pudq_inv(x_i), x_j)); //檢查一下

            double mu_i = z_ij(0) * x_j(0) + z_ij(1) * x_j(1);                      //檢查一下
            double omega_i = -z_ij(1) * x_j(0) + z_ij(0) * x_j(1);
            double eta_i = -z_ij(1) * x_j(0) + z_ij(0) * x_j(1);
            double kappa_i = -x_j(0) * z_ij(0) - x_j(1) * z_ij(1);
            double alpha_1 = -x_j(0) * z_ij(2) - x_j(1) * z_ij(3) + z_ij(0) * x_j(2) + z_ij(1) * x_j(3);
            double beta_1 = x_j(0) * z_ij(3) - x_j(1) * z_ij(2) - z_ij(1) * x_j(2) + z_ij(0) * x_j(3);
            double xi_1 = -x_j(0) * z_ij(0) + z_ij(1) * x_j(1);
            double zeta_1 = -z_ij(1) * x_j(0) - z_ij(0) * x_j(1);
            double alpha_3 = -x_j(0) * z_ij(3) + x_j(1) * z_ij(2) - x_j(2) * z_ij(1) + z_ij(0) * x_j(3);
            double beta_3 = -x_j(0) * z_ij(2) - x_j(1) * z_ij(3) - x_j(2) * z_ij(0) - z_ij(1) * x_j(3);
            
            double mu_j = x_i(0) * z_ij(0) - x_i(1) * z_ij(1);
            double omega_j = x_i(0) * z_ij(1) + x_i(1) * z_ij(0);
            double eta_j = -z_ij(1) * x_i(0) - z_ij(0) * x_i(1);
            double kappa_j = z_ij(0) * x_i(0) - z_ij(1) * x_i(1);
            double alpha_2 = -z_ij(2) * x_i(0) + z_ij(3) * x_i(1) - z_ij(0) * x_i(2) - z_ij(1) * x_i(3);
            double beta_2 = -z_ij(3) * x_i(0) - z_ij(2) * x_i(1) + z_ij(1) * x_i(2) - z_ij(0) * x_i(3);
            
            double phi = get_phi_atan2(r_ij(1), r_ij(0));
            double gamma = sinc1(phi);

            Eigen::Vector3d eij;
            eij << r_ij(1) / gamma, r_ij(2) / gamma, r_ij(3) / gamma;
            double f1 = f_1(phi);

            Eigen::Matrix<double, 3, 4> Aij = Eigen::Matrix<double, 3, 4>::Zero();
            Eigen::Matrix<double, 3, 4> Bij = Eigen::Matrix<double, 3, 4>::Zero();

            double dphi_dxi0 = eta_i * r_ij(0) - mu_i * r_ij(1);
            double dphi_dxi1 = kappa_i * r_ij(0) - omega_i * r_ij(1);

            Aij(0, 0) = eta_i / gamma + r_ij(1) * dphi_dxi0 * f1;
            Aij(0, 1) = kappa_i / gamma + r_ij(1) * dphi_dxi1 * f1;
            Aij(1, 0) = alpha_1 / gamma + r_ij(2) * dphi_dxi0 * f1;
            Aij(1, 1) = beta_1 / gamma + r_ij(2) * dphi_dxi1 * f1;
            Aij(1, 2) = xi_1 / gamma;
            Aij(1, 3) = zeta_1 / gamma;
            Aij(2, 0) = alpha_3 / gamma + r_ij(3) * dphi_dxi0 * f1;
            Aij(2, 1) = beta_3 / gamma + r_ij(3) * dphi_dxi1 * f1;
            Aij(2, 2) = -zeta_1 / gamma;
            Aij(2, 3) = xi_1 / gamma;

            double dphi_dxj0 = eta_j * r_ij(0) - mu_j * r_ij(1);
            double dphi_dxj1 = kappa_j * r_ij(0) - omega_j * r_ij(1);

            Bij(0, 0) = eta_j / gamma + r_ij(1) * dphi_dxj0 * f1;
            Bij(0, 1) = kappa_j / gamma + r_ij(1) * dphi_dxj1 * f1;
            Bij(1, 0) = alpha_2 / gamma + r_ij(2) * dphi_dxj0 * f1;
            Bij(1, 1) = beta_2 / gamma + r_ij(2) * dphi_dxj1 * f1;
            Bij(1, 2) = kappa_j / gamma;
            Bij(1, 3) = -eta_j / gamma;
            Bij(2, 0) = beta_2 / gamma + r_ij(3) * dphi_dxj0 * f1;
            Bij(2, 1) = -alpha_2 / gamma + r_ij(3) * dphi_dxj1 * f1;
            Bij(2, 2) = eta_j / gamma;
            Bij(2, 3) = kappa_j / gamma; // 以上都檢查一下

            int ij_index = 3 * ij;

            for (int i = 0; i < 3; i++){ 
                int e_row_idx = ij_index + i;
                E_triplets.emplace_back(e_row_idx, 0, eij(i));

                for (int j = 0; j < 4; j++){  
                    int i_index = 3 * ij + i;
                    int j_index_A = 4 * edge_i + j;
                    int j_index_B = 4 * edge_j + j;

                    J_triplets.emplace_back(i_index, j_index_A, Aij(i, j));
                    J_triplets.emplace_back(i_index, j_index_B, Bij(i, j));
                }
            }
        } 

        // Inter edges
        for (int ij = 0; ij < num_inter_edges; ij++){  
            bool edge_inverse = local_graph.inter_edges[ij].flag;
            int edge_i = local_graph.inter_edges[ij].i;
            int edge_j = local_graph.inter_edges[ij].j;

            int ij_index = 3 * (num_intra_edges + ij);

            if (!edge_inverse){
                PUDQ x_i = local_graph.vertices_pudq[edge_i];
                PUDQ z_ij = local_graph.inter_dp_pudq[ij];
                PUDQ x_j = local_graph.lm_vertices_pudq[edge_j];

                PUDQ r_ij = pudq_compose(pudq_inv(z_ij), pudq_mul(pudq_inv(x_i), x_j));
    
                double mu_i = z_ij(0) * x_j(0) + z_ij(1) * x_j(1);
                double omega_i = -z_ij(1) * x_j(0) + z_ij(0) * x_j(1);
                double eta_i = -z_ij(1) * x_j(0) + z_ij(0) * x_j(1);
                double kappa_i = -x_j(0) * z_ij(0) - x_j(1) * z_ij(1);
                double alpha_1 = -x_j(0) * z_ij(2) - x_j(1) * z_ij(3) + z_ij(0) * x_j(2) + z_ij(1) * x_j(3);
                double beta_1 = x_j(0) * z_ij(3) - x_j(1) * z_ij(2) - z_ij(1) * x_j(2) + z_ij(0) * x_j(3);
                double xi_1 = -x_j(0) * z_ij(0) + z_ij(1) * x_j(1);
                double zeta_1 = -z_ij(1) * x_j(0) - z_ij(0) * x_j(1);
                double alpha_3 = -x_j(0) * z_ij(3) + x_j(1) * z_ij(2) - x_j(2) * z_ij(1) + z_ij(0) * x_j(3);
                double beta_3 = -x_j(0) * z_ij(2) - x_j(1) * z_ij(3) - x_j(2) * z_ij(0) - z_ij(1) * x_j(3);
    
                double phi = get_phi_atan2(r_ij(1), r_ij(0));
                double gamma = sinc1(phi);
                Eigen::Vector3d eij;
                eij << r_ij(1) / gamma, r_ij(2) / gamma, r_ij(3) / gamma;
                double f1 = f_1(phi);
        
                Eigen::Matrix<double, 3, 4> Aij = Eigen::Matrix<double, 3, 4>::Zero();
                double dphi_dxi0 = eta_i * r_ij(0) - mu_i * r_ij(1);
                double dphi_dxi1 = kappa_i * r_ij(0) - omega_i * r_ij(1);
                Aij(0, 0) = eta_i / gamma + r_ij(1) * dphi_dxi0 * f1;
                Aij(0, 1) = kappa_i / gamma + r_ij(1) * dphi_dxi1 * f1;
                Aij(1, 0) = alpha_1 / gamma + r_ij(2) * dphi_dxi0 * f1;
                Aij(1, 1) = beta_1 / gamma + r_ij(2) * dphi_dxi1 * f1;
                Aij(1, 2) = xi_1 / gamma;
                Aij(1, 3) = zeta_1 / gamma;
                Aij(2, 0) = alpha_3 / gamma + r_ij(3) * dphi_dxi0 * f1;
                Aij(2, 1) = beta_3 / gamma + r_ij(3) * dphi_dxi1 * f1;
                Aij(2, 2) = -zeta_1 / gamma;
                Aij(2, 3) = xi_1 / gamma;
        
                for (int i = 0; i < 3; i++) {
                    int e_row_idx = ij_index + i;
                    E_triplets.emplace_back(e_row_idx, 0, eij(i));

                    for (int j = 0; j < 4; j++) {
                        int i_index = 3 * (num_intra_edges + ij) + i;
                        int j_index_A = 4 * edge_i + j;
    
                        J_triplets.emplace_back(i_index, j_index_A, Aij(i, j));
                    }
                }
            } else {
                PUDQ x_i = local_graph.lm_vertices_pudq[edge_i];
                PUDQ z_ij = local_graph.inter_dp_pudq[ij];
                PUDQ x_j = local_graph.vertices_pudq[edge_j];
        
                PUDQ r_ij = pudq_compose(pudq_inv(z_ij), pudq_mul(pudq_inv(x_i), x_j));
        
                double mu_j = x_i(0) * z_ij(0) - x_i(1) * z_ij(1);
                double omega_j = x_i(0) * z_ij(1) + x_i(1) * z_ij(0);
                double eta_j = -z_ij(1) * x_i(0) - z_ij(0) * x_i(1);
                double kappa_j = z_ij(0) * x_i(0) - z_ij(1) * x_i(1);
                double alpha_2 = -z_ij(2) * x_i(0) + z_ij(3) * x_i(1) - z_ij(0) * x_i(2) - z_ij(1) * x_i(3);
                double beta_2 = -z_ij(3) * x_i(0) - z_ij(2) * x_i(1) + z_ij(1) * x_i(2) - z_ij(0) * x_i(3);
                
                double phi = get_phi_atan2(r_ij(1), r_ij(0));
                double gamma = sinc1(phi);
                Eigen::Vector3d eij;
                eij << r_ij(1) / gamma, r_ij(2) / gamma, r_ij(3) / gamma;
                double f1 = f_1(phi);

                Eigen::Matrix<double, 3, 4> Bij = Eigen::Matrix<double, 3, 4>::Zero();
                double dphi_dxj0 = eta_j * r_ij(0) - mu_j * r_ij(1);
                double dphi_dxj1 = kappa_j * r_ij(0) - omega_j * r_ij(1);
                Bij(0, 0) = eta_j / gamma + r_ij(1) * dphi_dxj0 * f1;
                Bij(0, 1) = kappa_j / gamma + r_ij(1) * dphi_dxj1 * f1;
                Bij(1, 0) = alpha_2 / gamma + r_ij(2) * dphi_dxj0 * f1;
                Bij(1, 1) = beta_2 / gamma + r_ij(2) * dphi_dxj1 * f1;
                Bij(1, 2) = kappa_j / gamma;
                Bij(1, 3) = -eta_j / gamma;
                Bij(2, 0) = beta_2 / gamma + r_ij(3) * dphi_dxj0 * f1;
                Bij(2, 1) = -alpha_2 / gamma + r_ij(3) * dphi_dxj1 * f1;
                Bij(2, 2) = eta_j / gamma;
                Bij(2, 3) = kappa_j / gamma;

                for (int i = 0; i < 3; i++) {
                    int e_row_idx = ij_index + i;
                    E_triplets.emplace_back(e_row_idx, 0, eij(i));

                    for (int j = 0; j < 4; j++) {
                        int i_index = 3 * (num_intra_edges + ij) + i;
                        int j_index_B = 4 * edge_j + j;
        
                        J_triplets.emplace_back(i_index, j_index_B, Bij(i, j));
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> E(num_E_triplets, 1);
        E.setFromTriplets(E_triplets.begin(), E_triplets.end());

        Eigen::SparseMatrix<double> J_mat(num_E_triplets, 4 * N);
        J_mat.setFromTriplets(J_triplets.begin(), J_triplets.end());

        double F_X = 0.5 * Eigen::MatrixXd(E.transpose() * local_graph.Omega_ij * E)(0, 0);
        Eigen::SparseMatrix<double> egrad_X = J_mat.transpose() * local_graph.Omega_ij * E;
        Eigen::SparseMatrix<double> egnhess_X = J_mat.transpose() * local_graph.Omega_ij * J_mat;
        Eigen::SparseMatrix<double> P_X = P_X_N_sparse(local_graph.vertices_pudq);

        Eigen::VectorXd rgrad_X = Eigen::VectorXd(P_X * egrad_X);
        Eigen::SparseMatrix<double> rgnhess_X_sparse = P_X * egnhess_X * P_X;

        // Eigen::MatrixXd rgnhess_X = Eigen::MatrixXd(rgnhess_X_sparse);
        // rgnhess_X = rgnhess_X.triangularView<Lower>() + rgnhess_X.triangularView<StrictlyLower>().transpose();

        Result result;
        result.rgrad_X = rgrad_X;
        result.rgnhess_X = rgnhess_X_sparse;
        result.F_X = F_X;
        result.P_X = P_X;

        return result;
    }

    Result rgn_gradhess_J(const Graph& G) {
        const int N = static_cast<int>(G.vertices.size());
        const int M = static_cast<int>(G.edges.size());

        const int num_J_triplets = 24 * M; 
        std::vector<Eigen::Triplet<double>> J_triplets;
        J_triplets.reserve(num_J_triplets);

        const int num_E_triplets = 3 * M;
        std::vector<Eigen::Triplet<double>> E_triplets;
        E_triplets.reserve(num_E_triplets);

        for (int ij = 0; ij < M; ++ij) { 
            int edge_i = G.edges[ij].first;
            int edge_j = G.edges[ij].second;

            PUDQ x_i = G.vertices_pudq[edge_i];
            PUDQ x_j = G.vertices_pudq[edge_j];
            PUDQ z_ij = G.delta_poses_pudq[ij];

            
            PUDQ r_ij = pudq_compose(pudq_inv(z_ij), pudq_mul(pudq_inv(x_i), x_j));

            double mu_i = z_ij(0) * x_j(0) + z_ij(1) * x_j(1);
            double omega_i = -z_ij(1) * x_j(0) + z_ij(0) * x_j(1);
            double eta_i = -z_ij(1) * x_j(0) + z_ij(0) * x_j(1);
            double kappa_i = -x_j(0) * z_ij(0) - x_j(1) * z_ij(1);
            double alpha_1 = -x_j(0) * z_ij(2) - x_j(1) * z_ij(3) + z_ij(0) * x_j(2) + z_ij(1) * x_j(3);
            double beta_1 = x_j(0) * z_ij(3) - x_j(1) * z_ij(2) - z_ij(1) * x_j(2) + z_ij(0) * x_j(3);
            double xi_1 = -x_j(0) * z_ij(0) + z_ij(1) * x_j(1);
            double zeta_1 = -z_ij(1) * x_j(0) - z_ij(0) * x_j(1);
            double alpha_3 = -x_j(0) * z_ij(3) + x_j(1) * z_ij(2) - x_j(2) * z_ij(1) + z_ij(0) * x_j(3);
            double beta_3 = -x_j(0) * z_ij(2) - x_j(1) * z_ij(3) - x_j(2) * z_ij(0) - z_ij(1) * x_j(3);
            
            double mu_j = x_i(0) * z_ij(0) - x_i(1) * z_ij(1);
            double omega_j = x_i(0) * z_ij(1) + x_i(1) * z_ij(0);
            double eta_j = -z_ij(1) * x_i(0) - z_ij(0) * x_i(1);
            double kappa_j = z_ij(0) * x_i(0) - z_ij(1) * x_i(1);
            double alpha_2 = -z_ij(2) * x_i(0) + z_ij(3) * x_i(1) - z_ij(0) * x_i(2) - z_ij(1) * x_i(3);
            double beta_2 = -z_ij(3) * x_i(0) - z_ij(2) * x_i(1) + z_ij(1) * x_i(2) - z_ij(0) * x_i(3);
            
            double phi = get_phi_atan2(r_ij(1), r_ij(0));
            double gamma = sinc1(phi);

            Eigen::Vector3d eij;
            eij << r_ij(1) / gamma, r_ij(2) / gamma, r_ij(3) / gamma;
            double f1 = f_1(phi);

            Eigen::Matrix<double, 3, 4> Aij = Eigen::Matrix<double, 3, 4>::Zero();
            Eigen::Matrix<double, 3, 4> Bij = Eigen::Matrix<double, 3, 4>::Zero();

            double dphi_dxi0 = eta_i * r_ij(0) - mu_i * r_ij(1);
            double dphi_dxi1 = kappa_i * r_ij(0) - omega_i * r_ij(1);

            Aij(0, 0) = eta_i / gamma + r_ij(1) * dphi_dxi0 * f1;
            Aij(0, 1) = kappa_i / gamma + r_ij(1) * dphi_dxi1 * f1;
            Aij(1, 0) = alpha_1 / gamma + r_ij(2) * dphi_dxi0 * f1;
            Aij(1, 1) = beta_1 / gamma + r_ij(2) * dphi_dxi1 * f1;
            Aij(1, 2) = xi_1 / gamma;
            Aij(1, 3) = zeta_1 / gamma;
            Aij(2, 0) = alpha_3 / gamma + r_ij(3) * dphi_dxi0 * f1;
            Aij(2, 1) = beta_3 / gamma + r_ij(3) * dphi_dxi1 * f1;
            Aij(2, 2) = -zeta_1 / gamma;
            Aij(2, 3) = xi_1 / gamma;

            double dphi_dxj0 = eta_j * r_ij(0) - mu_j * r_ij(1);
            double dphi_dxj1 = kappa_j * r_ij(0) - omega_j * r_ij(1);

            Bij(0, 0) = eta_j / gamma + r_ij(1) * dphi_dxj0 * f1;
            Bij(0, 1) = kappa_j / gamma + r_ij(1) * dphi_dxj1 * f1;
            Bij(1, 0) = alpha_2 / gamma + r_ij(2) * dphi_dxj0 * f1;
            Bij(1, 1) = beta_2 / gamma + r_ij(2) * dphi_dxj1 * f1;
            Bij(1, 2) = kappa_j / gamma;
            Bij(1, 3) = -eta_j / gamma;
            Bij(2, 0) = beta_2 / gamma + r_ij(3) * dphi_dxj0 * f1;
            Bij(2, 1) = -alpha_2 / gamma + r_ij(3) * dphi_dxj1 * f1;
            Bij(2, 2) = eta_j / gamma;
            Bij(2, 3) = kappa_j / gamma;

            int ij_index = 3 * ij;

            for (int i = 0; i < 3; i++) { 
                int e_row_idx = ij_index + i;
                E_triplets.emplace_back(e_row_idx, 0, eij(i));

                for (int j = 0; j < 4; j++) {  
                    int i_index = 3 * ij + i;
                    int j_index_A = 4 * edge_i + j;
                    int j_index_B = 4 * edge_j + j;

                    J_triplets.emplace_back(i_index, j_index_A, Aij(i, j));
                    J_triplets.emplace_back(i_index, j_index_B, Bij(i, j));
                }
            }
        } 

        Eigen::SparseMatrix<double> E(num_E_triplets, 1);
        E.setFromTriplets(E_triplets.begin(), E_triplets.end());

        Eigen::SparseMatrix<double> J_mat(num_E_triplets, 4 * N);
        J_mat.setFromTriplets(J_triplets.begin(), J_triplets.end());

        double F_X = 0.5 * Eigen::MatrixXd(E.transpose() * G.Omega * E)(0, 0);
        Eigen::SparseMatrix<double> egrad_X = J_mat.transpose() * G.Omega * E;
        Eigen::SparseMatrix<double> egnhess_X = J_mat.transpose() * G.Omega * J_mat;

        Eigen::SparseMatrix<double> P_X = P_X_N_sparse(G.vertices_pudq);

        Eigen::VectorXd rgrad_X = Eigen::VectorXd(P_X * egrad_X);
        Eigen::SparseMatrix<double> rgnhess_X_sparse = P_X * egnhess_X * P_X;

        Result result;
        result.rgrad_X = rgrad_X;
        result.rgnhess_X = rgnhess_X_sparse;
        result.F_X = F_X;
        result.P_X = P_X;

        return result;
    }

    Eigen::VectorXd G_get_X(const LocalGraph& G){
        const int N = static_cast<int>(G.vertices_pudq.size());
        Eigen::VectorXd X(4 * N);
        for (int i = 0; i < N; ++i) { //確認一下這裡的index是否正確
            X.segment<4>(4 * i) = G.vertices_pudq[i];  // 搜尋一下這個對不對
        }
        return X;
    }

    LocalGraph G_set_X(LocalGraph& G, const Eigen::VectorXd& X_new){
        const int N = static_cast<int>(G.vertices_pudq.size());

        if (X_new.size() != 4 * N) throw std::invalid_argument("G_set_X: BUG");

        for (int i = 0; i < N; ++i) {
            G.vertices_pudq[i] = X_new.segment<4>(4 * i);  // 搜尋一下這個對不對
            G.vertices[i]      = pudqlib::pudq_to_pose(G.vertices_pudq[i]);
        }
        return G;
    }

}