#include "pudq_lib.hpp"
#include "graph.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <filesystem>
#include <climits>


namespace graph{

    pudqlib::Graph load_g2o_file_pudq(const std::string& g2o_file){
        std::ifstream fin(g2o_file);
        if (!fin) throw std::runtime_error("Fail opening g2o file: " + g2o_file);

        std::unordered_map<int, Eigen::Vector3d> pose_temp;
        std::vector<std::pair<int, int>> edges_temp;
        std::vector<Eigen::Vector3d> dp_temp;
        std::vector<Eigen::Vector2d> t_temp;
        std::vector<Eigen::Matrix2d> R_temp;
        std::vector<Eigen::Matrix3d> info_temp;
        std::vector<double>          tau_temp;
        std::vector<double>          kappa_temp;

        int max_vertex_id = -1;
        std::string line;
        std::size_t line_count = 0;

        while (std::getline(fin, line)){
            ++line_count;
            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "VERTEX_SE2"){
                int id;
                double x, y, th;
                if (!(iss >> id >> x >> y >> th))
                    throw std::runtime_error("Weird VERTEX_SE2 at line " + std::to_string(line_count));
                max_vertex_id = std::max(max_vertex_id, id);
                pose_temp[id] = Eigen::Vector3d(x, y, th);

            } else if (token == "EDGE_SE2"){
                int id1, id2;
                double dx, dy, dth, I11, I12, I13, I22, I23, I33;
                if (!(iss >> id1 >> id2 >> dx >> dy >> dth >> I11 >> I12 >> I13 >> I22 >> I23 >> I33))
                    throw std::runtime_error("Weird EDGE_SE2 at line " + std::to_string(line_count));

                edges_temp.emplace_back(id1, id2);
                dp_temp.emplace_back(dx, dy, dth);
                t_temp.emplace_back(dx, dy);
                R_temp.emplace_back(pudqlib::R_from_theta(dth));

                Eigen::Matrix3d measurement_info;
                measurement_info << I11, I12, I13,
                                    I12, I22, I23,
                                    I13, I23, I33;
                info_temp.emplace_back(measurement_info);

                Eigen::Matrix2d ta = measurement_info.block<2, 2>(0,0);
                double ta_inv = ta.inverse().trace();
                tau_temp.emplace_back(2 / ta_inv);
                kappa_temp.emplace_back(I33);
            }

            if (line_count % 1000 == 0){
                std::cout << "Reading line " << line_count << "\n";
            }
        }

        pudqlib::Graph G;
        const int N = max_vertex_id + 1;

        G.vertices_true.resize(N, Eigen::Vector3d::Zero());
        G.vertices_pudq_true.resize(N, pudqlib::PUDQ::Zero());
        for (const auto& mapvalue : pose_temp){
            const int id = mapvalue.first;
            const Eigen::Vector3d& pose = mapvalue.second;
            G.vertices_true[id] = pose;
            G.vertices_pudq_true[id] = pudqlib::pose_to_pudq(pose);
        }

        G.edges = edges_temp;
        G.delta_poses = dp_temp;
        G.t = t_temp;
        G.R = R_temp;
        G.information = info_temp;
        G.tau = tau_temp;
        G.kappa =  kappa_temp;

        G.information_pudq.reserve(info_temp.size());
        G.information_se2.reserve(info_temp.size());
        G.delta_poses_pudq.reserve(dp_temp.size());
        for (std::size_t e_idx = 0; e_idx < info_temp.size(); ++e_idx){
            const Eigen::Matrix3d info_p = pudqlib::info_eucl_to_pudq(info_temp[e_idx], 0.0);
            const Eigen::Matrix3d info_s = pudqlib::info_eucl_to_se2(info_temp[e_idx], 0.0);
            G.information_pudq.push_back(info_p);
            G.information_se2.push_back(info_s);
            G.delta_poses_pudq.push_back(pudqlib::pose_to_pudq(dp_temp[e_idx]));
        }

        std::cout << "Done reading, now computing odometry vertices\n";
        G.vertices.resize(N);
        G.vertices_pudq.resize(N);

        G.vertices[0] = Eigen::Vector3d::Zero();
        G.vertices_pudq[0] = pudqlib::pose_to_pudq(G.vertices[0]);

        for (int i = 1; i < N; ++i){
            std::size_t edge_index = static_cast<std::size_t>(-1);
            for (std::size_t e = 0; e < G.edges.size(); ++e){
                if (G.edges[e].first == (i - 1) && G.edges[e].second == i){
                    edge_index = e;
                    break;
                }
            }
            
            if (edge_index == static_cast<std::size_t>(-1)) {
            throw std::runtime_error("No edge found between vertices " + std::to_string(i-1) + " and " + std::to_string(i));
            }

            const Eigen::Vector2d& t_ij = t_temp[edge_index];
            const Eigen::Matrix2d& R_ij = R_temp[edge_index];

            const Eigen::Vector3d& vi = G.vertices[i - 1];
            const Eigen::Vector2d  t_i(vi.x(), vi.y());
            const double theta_i = vi.z();
            const Eigen::Matrix2d R_i = pudqlib::R_from_theta(theta_i);

            const Eigen::Vector2d  t_j = t_i + R_i * t_ij;
            const Eigen::Matrix2d  R_j = R_i * R_ij;
            const double theta_j = pudqlib::theta_from_R(R_j);

            G.vertices[i] = Eigen::Vector3d(t_j.x(), t_j.y(), theta_j);
            G.vertices_pudq[i] = pudqlib::pose_to_pudq(G.vertices[i]);
        }

        std::cout << "Done computing odometry vertices\n";
        G.Omega = pudqlib::Omega_sparse(G);
        std::cout << "Finished Data Loading\n";
        return G;
    }

    pudqlib::Graph load_g2o_file_pudq_V2(const std::string& g2o_file){
        std::ifstream fin(g2o_file);
        if (!fin) throw std::runtime_error("Fail opening g2o file: " + g2o_file);

        std::unordered_map<int, Eigen::Vector3d> pose_temp;
        std::vector<std::pair<int, int>> edges_temp;
        std::vector<Eigen::Vector3d> dp_temp;
        std::vector<Eigen::Vector2d> t_temp;
        std::vector<Eigen::Matrix2d> R_temp;
        std::vector<Eigen::Matrix3d> info_temp;
        std::vector<double>          tau_temp;
        std::vector<double>          kappa_temp;

        int max_vertex_id = -1;
        std::string line;
        std::size_t line_count = 0;

        while (std::getline(fin, line)){
            ++line_count;
            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "VERTEX_SE2"){
                int id;
                double x, y, th;
                if (!(iss >> id >> x >> y >> th))
                    throw std::runtime_error("Weird VERTEX_SE2 at line " + std::to_string(line_count));
                max_vertex_id = std::max(max_vertex_id, id);
                pose_temp[id] = Eigen::Vector3d(x, y, th);

            } else if (token == "EDGE_SE2"){
                int id1, id2;
                double dx, dy, dth, I11, I12, I13, I22, I23, I33;
                if (!(iss >> id1 >> id2 >> dx >> dy >> dth >> I11 >> I12 >> I13 >> I22 >> I23 >> I33))
                    throw std::runtime_error("Weird EDGE_SE2 at line " + std::to_string(line_count));

                edges_temp.emplace_back(id1, id2);
                dp_temp.emplace_back(dx, dy, dth);
                t_temp.emplace_back(dx, dy);
                R_temp.emplace_back(pudqlib::R_from_theta(dth));

                Eigen::Matrix3d measurement_info;
                measurement_info << I11, I12, I13,
                                    I12, I22, I23,
                                    I13, I23, I33;
                info_temp.emplace_back(measurement_info);

                Eigen::Matrix2d ta = measurement_info.block<2, 2>(0,0);
                double ta_inv = ta.inverse().trace();
                tau_temp.emplace_back(2 / ta_inv);
                kappa_temp.emplace_back(I33);
            }

            if (line_count % 1000 == 0){
                std::cout << "Reading line " << line_count << "\n";
            }
        }

        pudqlib::Graph G;
        const int N = max_vertex_id + 1;

        G.vertices_true.resize(N, Eigen::Vector3d::Zero());
        G.vertices_pudq_true.resize(N, pudqlib::PUDQ::Zero());
        G.vertices.resize(N);
        G.vertices_pudq.resize(N);

        for (const auto& mapvalue : pose_temp){
            const int id = mapvalue.first;
            const Eigen::Vector3d& pose = mapvalue.second;
            G.vertices_true[id] = pose;
            G.vertices_pudq_true[id] = pudqlib::pose_to_pudq(pose);
            G.vertices[id] = pose;                       
            G.vertices_pudq[id] = pudqlib::pose_to_pudq(pose);
        }

        G.edges = edges_temp;
        G.delta_poses = dp_temp;
        G.t = t_temp;
        G.R = R_temp;
        G.information = info_temp;
        G.tau = tau_temp;
        G.kappa =  kappa_temp;

        G.information_pudq.reserve(info_temp.size());
        G.information_se2.reserve(info_temp.size());
        G.delta_poses_pudq.reserve(dp_temp.size());
        for (std::size_t e_idx = 0; e_idx < info_temp.size(); ++e_idx){
            const Eigen::Matrix3d info_p = pudqlib::info_eucl_to_pudq(info_temp[e_idx], 0.0);
            const Eigen::Matrix3d info_s = pudqlib::info_eucl_to_se2(info_temp[e_idx], 0.0);
            G.information_pudq.push_back(info_p);
            G.information_se2.push_back(info_s);
            G.delta_poses_pudq.push_back(pudqlib::pose_to_pudq(dp_temp[e_idx]));
        }

        G.Omega = pudqlib::Omega_sparse(G);
        std::cout << "Finished Data Loading\n";
        return G;
    }


    pudqlib::Graph load_g2o_file_pudq_gt(const std::string& g2o_file){
        std::ifstream fin(g2o_file);
        if (!fin) throw std::runtime_error("Fail opening g2o file: " + g2o_file);

        std::unordered_map<int, Eigen::Vector3d> pose_temp;
        std::vector<std::pair<int, int>> edges_temp;
        std::vector<Eigen::Vector3d> dp_temp;
        std::vector<Eigen::Vector2d> t_temp;
        std::vector<Eigen::Matrix2d> R_temp;
        std::vector<Eigen::Matrix3d> info_temp;
        std::vector<double>          tau_temp;
        std::vector<double>          kappa_temp;

        int max_vertex_id = -1;
        std::string line;
        std::size_t line_count = 0;

        while (std::getline(fin, line)){
            ++line_count;
            if (line.empty()) continue;

            std::istringstream iss(line);
            std::string token;
            iss >> token;

            if (token == "VERTEX_SE2"){
                int id;
                double x, y, th;
                if (!(iss >> id >> x >> y >> th))
                    throw std::runtime_error("Weird VERTEX_SE2 at line " + std::to_string(line_count));
                max_vertex_id = std::max(max_vertex_id, id);
                pose_temp[id] = Eigen::Vector3d(x, y, th);
                
            } else if (token == "EDGE_SE2"){
                int id1, id2;
                double dx, dy, dth, I11, I12, I13, I22, I23, I33;
                if (!(iss >> id1 >> id2 >> dx >> dy >> dth >> I11 >> I12 >> I13 >> I22 >> I23 >> I33))
                    throw std::runtime_error("Weird EDGE_SE2 at line " + std::to_string(line_count));

                edges_temp.emplace_back(id1, id2);
                dp_temp.emplace_back(dx, dy, dth);
                t_temp.emplace_back(dx, dy);
                R_temp.emplace_back(pudqlib::R_from_theta(dth));

                Eigen::Matrix3d measurement_info;
                measurement_info << I11, I12, I13,
                                    I12, I22, I23,
                                    I13, I23, I33;
                info_temp.emplace_back(measurement_info);

                Eigen::Matrix2d ta = measurement_info.block<2, 2>(0,0);
                double ta_inv = ta.inverse().trace();
                tau_temp.emplace_back(2 / ta_inv);
                kappa_temp.emplace_back(I33);
            }

            if (line_count % 1000 == 0){
                std::cout << "Reading line " << line_count << "\n";
            }
        }

        pudqlib::Graph G;
        const int N = max_vertex_id + 1;

        G.vertices.resize(N);
        G.vertices_pudq.resize(N);
        G.vertices_true.resize(N, Eigen::Vector3d::Zero());
        G.vertices_pudq_true.resize(N, pudqlib::PUDQ::Zero());

        for (int i = 0; i < N; ++i){
            auto it = pose_temp.find(i);
            if (it == pose_temp.end()) {
                throw std::runtime_error("Ground truth vertex " + std::to_string(i) + " is missing!");
            }
            const Eigen::Vector3d& v = it->second;
            G.vertices[i] = v;                        
            G.vertices_pudq[i] = pudqlib::pose_to_pudq(v); 
            G.vertices_true[i] = v;                       
            G.vertices_pudq_true[i] = pudqlib::pose_to_pudq(v);
        }
        std::cout << "Done forming GT vertices\n";

        G.edges = edges_temp;
        G.delta_poses = dp_temp;
        G.t = t_temp;
        G.R = R_temp;
        G.information = info_temp;
        G.tau = tau_temp;
        G.kappa =  kappa_temp;

        G.information_pudq.reserve(info_temp.size());
        G.information_se2.reserve(info_temp.size());
        G.delta_poses_pudq.reserve(dp_temp.size());
        for (std::size_t e_idx = 0; e_idx < info_temp.size(); ++e_idx){
            const Eigen::Matrix3d info_p = pudqlib::info_eucl_to_pudq(info_temp[e_idx], 0.0);
            const Eigen::Matrix3d info_s = pudqlib::info_eucl_to_se2(info_temp[e_idx], 0.0);
            G.information_pudq.push_back(info_p);
            G.information_se2.push_back(info_s);
            G.delta_poses_pudq.push_back(pudqlib::pose_to_pudq(dp_temp[e_idx]));
        }

        G.Omega = pudqlib::Omega_sparse(G);
        std::cout << "Finished GT Data Loading\n";
        return G;
    }

    pudqlib::Graph combine_gt_odom_g2o(const pudqlib::Graph& G_odom, const pudqlib::Graph& G_gt){
        pudqlib::Graph graph_final = G_odom;

        graph_final.vertices_true = G_gt.vertices_true;
        graph_final.vertices_pudq_true = G_gt.vertices_pudq_true;
        graph_final.delta_poses_true = G_gt.delta_poses;
        graph_final.delta_poses_pudq_true = G_gt.delta_poses_pudq;

        const std::size_t M = graph_final.edges.size();
        if (G_gt.edges.size() != M || G_gt.delta_poses_pudq.size() != M || graph_final.delta_poses_pudq.size() != M || graph_final.information.size() != M) {
            throw std::runtime_error("combine_gt_odom_g2o: mismatched edge/measurement sizes.");
        }

        for (std::size_t ij = 0; ij < M; ++ij){
            const pudqlib::PUDQ& z_ij_true = G_gt.delta_poses_pudq[ij];
            const pudqlib::PUDQ& z_ij_noisy = graph_final.delta_poses_pudq[ij];
            const pudqlib::PUDQ exp_eta_ij = pudqlib::pudq_compose(pudqlib::pudq_inv(z_ij_true), z_ij_noisy);
            const Eigen::Vector3d eta_ij = pudqlib::Log_1(exp_eta_ij);

            const Eigen::Matrix3d& information_euc = graph_final.information[ij];
            const Eigen::Matrix3d information_pudq = pudqlib::info_eucl_to_pudq(information_euc, 2.0 * eta_ij(0));
            const Eigen::Matrix3d information_se2 = pudqlib::info_pudq_to_se2(information_pudq);

            graph_final.information_pudq[ij] = information_pudq;
            graph_final.information_se2[ij]  = information_se2;
        }

        graph_final.Omega = pudqlib::Omega_sparse(graph_final);
        return graph_final;
    }

    int find_local_vertex_index (const pudqlib::MultiGraph& rg, int g_index){
        const auto& vertices_in = rg.vertices_interval; 
        auto it = std::find(vertices_in.begin(), vertices_in.end(), g_index); 
        if (it == vertices_in.end()){ 
            throw std::runtime_error("Global vertex " + std::to_string(g_index) + " not found in robot mapping");
        }
        return static_cast<int>(std::distance(vertices_in.begin(), it)); 
    }

    int find_robot_for_vertex (const std::vector<pudqlib::MultiGraph>& mg, int num_robots, int g_vertex){
        for (int r = 0; r < num_robots; ++r){
            const auto& vertices_in = mg[r].vertices_interval;
            if (std::find(vertices_in.begin(), vertices_in.end(), g_vertex) != vertices_in.end()){ 
                return r;
            }
        }
        throw std::runtime_error("Vertex " + std::to_string(g_vertex) + " not found in any robot");
    }

    std::pair<int,bool> findOrAddLandmark(pudqlib::MultiGraph& rg, int r_j, int v_j_local){
        for (int lm = 0; lm < static_cast<int>(rg.lm_foreign_info.size()); ++lm){
            const auto& info = rg.lm_foreign_info[lm];
            if (info.robot == r_j && info.local_index == v_j_local){
                return {lm, true};
            }
        }
        int Lm_idx = static_cast<int>(rg.lm_foreign_info.size());
        return {Lm_idx, false};
    }

    std::vector<pudqlib::MultiGraph> distributebiggraph_withinverseedge(const pudqlib::Graph& graph, int num_robots){
        std::vector<pudqlib::MultiGraph> multi_graph(num_robots); 

        const int vertices_total = static_cast<int>(graph.vertices.size());
        const int vertices_per_robot = vertices_total / num_robots; 

        for (int r = 0; r < num_robots; ++r){
            const int start_idx = r * vertices_per_robot; 
            const int end_idx   = (r == num_robots - 1) ? (vertices_total - 1)
                                         : ((r + 1) * vertices_per_robot - 1);
            const int vertices_num = end_idx - start_idx + 1;

            auto& mulg = multi_graph[r];   
            mulg.vertices_true.resize(vertices_num);
            mulg.vertices_pudq_true.resize(vertices_num);
            mulg.vertices.resize(vertices_num);
            mulg.vertices_pudq.resize(vertices_num);
            mulg.vertices_interval.resize(vertices_num);

            for (int i = 0; i < vertices_num; ++i){
                const int global_vertex = start_idx + i; 
                mulg.vertices_true[i] = graph.vertices_true[global_vertex];
                mulg.vertices_pudq_true[i] = graph.vertices_pudq_true[global_vertex];
                mulg.vertices[i] = graph.vertices[global_vertex];
                mulg.vertices_pudq[i] = graph.vertices_pudq[global_vertex];
                mulg.vertices_interval[i] = global_vertex;  
            } 

            mulg.intra_edges.clear();
            mulg.intra_dp.clear();             
            mulg.intra_dp_true.clear();
            mulg.intra_dp_pudq.clear();        
            mulg.intra_dp_pudq_true.clear();
            mulg.intra_t.clear();              
            mulg.intra_R.clear();
            mulg.intra_info.clear();           
            mulg.intra_info_pudq.clear();
            mulg.intra_info_se2.clear();       
            mulg.intra_tau.clear();
            mulg.intra_kappa.clear();

            mulg.inter_edges.clear();
            mulg.inter_dp.clear();             
            mulg.inter_dp_true.clear();
            mulg.inter_dp_pudq.clear();        
            mulg.inter_dp_pudq_true.clear();
            mulg.inter_t.clear();              
            mulg.inter_R.clear();
            mulg.inter_info.clear();           
            mulg.inter_info_pudq.clear();
            mulg.inter_info_se2.clear();       
            mulg.inter_tau.clear();
            mulg.inter_kappa.clear();

            mulg.lm_vertices.clear();          
            mulg.lm_vertices_pudq.clear();
            mulg.lm_vertices_true.clear();     
            mulg.lm_vertices_pudq_true.clear();
            mulg.lm_foreign_info.clear();
        }

        for (std::size_t e = 0; e < graph.edges.size(); ++e){
            const int v_i = graph.edges[e].first;
            const int v_j = graph.edges[e].second;

            const int r_i = find_robot_for_vertex(multi_graph, num_robots, v_i);
            const int r_j = find_robot_for_vertex(multi_graph, num_robots, v_j);

            if (r_i == r_j){
                auto& mulg_intra = multi_graph[r_i]; 
                const int v_i_local = find_local_vertex_index(mulg_intra, v_i);
                const int v_j_local = find_local_vertex_index(mulg_intra, v_j);

                mulg_intra.intra_edges.emplace_back(Eigen::Vector2i(v_i_local, v_j_local));
                mulg_intra.intra_dp.push_back(graph.delta_poses[e]);
                mulg_intra.intra_dp_true.push_back(graph.delta_poses_true[e]);
                mulg_intra.intra_dp_pudq.push_back(graph.delta_poses_pudq[e]);
                mulg_intra.intra_dp_pudq_true.push_back(graph.delta_poses_pudq_true[e]);
                mulg_intra.intra_t.push_back(graph.t[e]);
                mulg_intra.intra_R.push_back(graph.R[e]);
                mulg_intra.intra_info.push_back(graph.information[e]);
                mulg_intra.intra_info_pudq.push_back(graph.information_pudq[e]);
                mulg_intra.intra_info_se2.push_back(graph.information_se2[e]);
                mulg_intra.intra_tau.push_back(graph.tau[e]);
                mulg_intra.intra_kappa.push_back(graph.kappa[e]);
            } else{
                auto& mulg_i = multi_graph[r_i];  
                auto& mulg_j = multi_graph[r_j];  

                const int v_i_local = find_local_vertex_index(mulg_i, v_i);
                const int v_j_local = find_local_vertex_index(mulg_j, v_j);

                auto [Lm_idx_1, Exists_1] = findOrAddLandmark(mulg_i, r_j, v_j_local);
                if (!Exists_1) {
                    mulg_i.lm_vertices.push_back(graph.vertices[v_j]);
                    mulg_i.lm_vertices_pudq.push_back(graph.vertices_pudq[v_j]);
                    mulg_i.lm_vertices_true.push_back(graph.vertices_true[v_j]);
                    mulg_i.lm_vertices_pudq_true.push_back(graph.vertices_pudq_true[v_j]);
                    mulg_i.lm_foreign_info.push_back(pudqlib::ForeignInfo{r_j, v_j_local, v_j});
                }

                auto [Lm_idx_2, Exists_2] = findOrAddLandmark(mulg_j, r_i, v_i_local);
                if (!Exists_2) {
                    mulg_j.lm_vertices.push_back(graph.vertices[v_i]);
                    mulg_j.lm_vertices_pudq.push_back(graph.vertices_pudq[v_i]);
                    mulg_j.lm_vertices_true.push_back(graph.vertices_true[v_i]);
                    mulg_j.lm_vertices_pudq_true.push_back(graph.vertices_pudq_true[v_i]);
                    mulg_j.lm_foreign_info.push_back(pudqlib::ForeignInfo{r_i, v_i_local, v_i});
                }

                mulg_i.inter_edges.push_back(pudqlib::InterEdge{ v_i_local, Lm_idx_1, false });
                mulg_i.inter_dp.push_back(graph.delta_poses[e]);
                mulg_i.inter_dp_true.push_back(graph.delta_poses_true[e]);
                mulg_i.inter_dp_pudq.push_back(graph.delta_poses_pudq[e]);
                mulg_i.inter_dp_pudq_true.push_back(graph.delta_poses_pudq_true[e]);
                mulg_i.inter_t.push_back(graph.t[e]);
                mulg_i.inter_R.push_back(graph.R[e]);
                mulg_i.inter_info.push_back(graph.information[e]);
                mulg_i.inter_info_pudq.push_back(graph.information_pudq[e]);
                mulg_i.inter_info_se2.push_back(graph.information_se2[e]);
                mulg_i.inter_tau.push_back(graph.tau[e]);
                mulg_i.inter_kappa.push_back(graph.kappa[e]);

                mulg_j.inter_edges.push_back(pudqlib::InterEdge{ Lm_idx_2, v_j_local, true });
                mulg_j.inter_dp.push_back(graph.delta_poses[e]);
                mulg_j.inter_dp_true.push_back(graph.delta_poses_true[e]);
                mulg_j.inter_dp_pudq.push_back(graph.delta_poses_pudq[e]);
                mulg_j.inter_dp_pudq_true.push_back(graph.delta_poses_pudq_true[e]);
                mulg_j.inter_t.push_back(graph.t[e]);
                mulg_j.inter_R.push_back(graph.R[e]);
                mulg_j.inter_info.push_back(graph.information[e]);
                mulg_j.inter_info_pudq.push_back(graph.information_pudq[e]);
                mulg_j.inter_info_se2.push_back(graph.information_se2[e]);
                mulg_j.inter_tau.push_back(graph.tau[e]);
                mulg_j.inter_kappa.push_back(graph.kappa[e]);
            }
        }

        for (int r = 0; r < num_robots; ++r) {
            multi_graph[r].Omega_ij = Omega_sparse_Multi(multi_graph[r]);
        }

        std::cout << "Successfully distributed gridworld\n";
        return multi_graph;
    }

    pudqlib::LocalGraph build_local_subgraph(const std::vector<pudqlib::MultiGraph>& multi_graph, int r){
        if (r < 0 || r >= static_cast<int>(multi_graph.size())) 
            throw std::out_of_range("build_local_subgraph: robot index out of range");

        const pudqlib::MultiGraph& srcdata = multi_graph[r];
        pudqlib::LocalGraph lg;

        lg.robot_id     = r;         
        lg.anchor_first = (r == 0);   

        lg.vertices_true        = srcdata.vertices_true;
        lg.vertices_pudq_true   = srcdata.vertices_pudq_true;
        lg.vertices             = srcdata.vertices;
        lg.vertices_pudq        = srcdata.vertices_pudq;
        lg.vertices_interval    = srcdata.vertices_interval; 

        lg.lm_vertices          = srcdata.lm_vertices;
        lg.lm_vertices_pudq     = srcdata.lm_vertices_pudq;
        lg.lm_vertices_true     = srcdata.lm_vertices_true;
        lg.lm_vertices_pudq_true= srcdata.lm_vertices_pudq_true;
        lg.lm_foreign_info      = srcdata.lm_foreign_info;

        lg.intra_edges          = srcdata.intra_edges;
        lg.intra_dp             = srcdata.intra_dp;
        lg.intra_dp_true        = srcdata.intra_dp_true;
        lg.intra_dp_pudq        = srcdata.intra_dp_pudq;
        lg.intra_dp_pudq_true   = srcdata.intra_dp_pudq_true;
        lg.intra_t              = srcdata.intra_t;
        lg.intra_R              = srcdata.intra_R;
        lg.intra_info           = srcdata.intra_info;
        lg.intra_info_pudq      = srcdata.intra_info_pudq;
        lg.intra_info_se2       = srcdata.intra_info_se2;
        lg.intra_tau            = srcdata.intra_tau;
        lg.intra_kappa          = srcdata.intra_kappa;

        lg.inter_edges          = srcdata.inter_edges;
        lg.inter_dp             = srcdata.inter_dp;
        lg.inter_dp_true        = srcdata.inter_dp_true;
        lg.inter_dp_pudq        = srcdata.inter_dp_pudq;
        lg.inter_dp_pudq_true   = srcdata.inter_dp_pudq_true;
        lg.inter_t              = srcdata.inter_t;
        lg.inter_R              = srcdata.inter_R;
        lg.inter_info           = srcdata.inter_info;
        lg.inter_info_pudq      = srcdata.inter_info_pudq;
        lg.inter_info_se2       = srcdata.inter_info_se2;
        lg.inter_tau            = srcdata.inter_tau;
        lg.inter_kappa          = srcdata.inter_kappa;

        lg.Omega_ij             = srcdata.Omega_ij; 

        return lg;
    }

    pudqlib::Graph reconstructglobalgraph(const std::vector<pudqlib::Robot>& robots, const pudqlib::Graph& original_G
    ) {
        const int num_robots = static_cast<int>(robots.size());
        const int total_vertices = static_cast<int>(original_G.vertices.size());

        std::cout << "Reconstructing global graph from " << num_robots 
                  << " robots (" << total_vertices << " total vertices)..." << std::endl;

        pudqlib::Graph G_optimized = original_G;

        if (G_optimized.vertices.size() != total_vertices) {
            G_optimized.vertices.resize(total_vertices);
        }
        if (G_optimized.vertices_pudq.size() != total_vertices) {
            G_optimized.vertices_pudq.resize(total_vertices);
        }

        for (int r = 0; r < num_robots; ++r) {
            const pudqlib::LocalGraph& local_graph = robots[r].local_graph;
            const std::vector<int>& global_mapping = local_graph.vertices_interval;
            const int num_local_vertices = static_cast<int>(local_graph.vertices_pudq.size());

            int min_global = (global_mapping.empty()) ? -1 : *std::min_element(global_mapping.begin(), global_mapping.end());
            int max_global = (global_mapping.empty()) ? -1 : *std::max_element(global_mapping.begin(), global_mapping.end());
            
            std::printf("Robot %d: processing %d local vertices (global indices %d to %d)\n", r, num_local_vertices, min_global, max_global);

            for (int local_idx = 0; local_idx < num_local_vertices; ++local_idx) {

                if (local_idx >= static_cast<int>(global_mapping.size())) {
                     std::cerr << "Warning: local_idx " << local_idx 
                               << " out of bounds for global_mapping on robot " << r << std::endl;
                     continue; 
                }

                int global_idx = global_mapping[local_idx]; 

                if (global_idx >= 0 && global_idx < total_vertices) {
                    const pudqlib::PUDQ& optimized_pudq = local_graph.vertices_pudq[local_idx];
                    G_optimized.vertices_pudq[global_idx] = optimized_pudq;
                    G_optimized.vertices[global_idx] = pudqlib::pudq_to_pose(optimized_pudq);

                } else {
                    std::cerr << "Warning: Invalid global_idx " << global_idx 
                              << " from robot " << r << ", local_idx " << local_idx << std::endl;
                }
            }
        }

        std::cout << "Global graph reconstruction complete." << std::endl;
        return G_optimized;
    }

    Eigen::MatrixXd readMatrixFromFile(const std::string& filename) {
        std::vector<std::vector<double>> v;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::string line;
        std::size_t max_cols = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            v.emplace_back();
            std::istringstream iss(line);
            double val;
            while (iss >> val) {
                v.back().push_back(val);
            }
            if (v.back().empty()) {
                v.pop_back(); 
            } else if (v.back().size() > max_cols) {
                max_cols = v.back().size();
            }
        }
        
        if (v.empty()) {
            return Eigen::MatrixXd(0, 0);
        }

        Eigen::MatrixXd mat(v.size(), max_cols);
        for (std::size_t i = 0; i < v.size(); ++i) {
            for (std::size_t j = 0; j < v[i].size(); ++j) {
                mat(i, j) = v[i][j];
            }
            for (std::size_t j = v[i].size(); j < max_cols; ++j) {
                 mat(i, j) = 0.0;
            }
        }
        return mat;
    }
    
    pudqlib::Graph load_graph_from_matlab_export(const std::string& directory_path) {
        namespace fs = std::filesystem;

        fs::path dir(directory_path);
        if (!fs::exists(dir)) {
            throw std::runtime_error("Directory not found: " + dir.string());
        }

        auto need = [&](const char* name) -> fs::path {
            fs::path p = dir / name;
            if (!fs::exists(p)) {
                throw std::runtime_error(std::string("Missing file: ") + p.string());
            }
            return p;
        };

        std::cout << "Loading pre-initialized graph from MATLAB export: " << dir.string() << std::endl;
        pudqlib::Graph G;

        // Vertices 
        Eigen::MatrixXd vertices_pudq_mat = readMatrixFromFile(need("vertices_pudq.txt"));
        const int N = static_cast<int>(vertices_pudq_mat.rows());
        if (N <= 0) throw std::runtime_error("No vertices found in vertices_pudq.txt");

        G.vertices_pudq.resize(N);
        G.vertices.resize(N);
        for (int i = 0; i < N; ++i) {
            G.vertices_pudq[i] = vertices_pudq_mat.row(i).transpose();
            G.vertices[i]      = pudqlib::pudq_to_pose(G.vertices_pudq[i]);
        }
        std::cout << "Loaded " << N << " vertices (from vertices_pudq.txt)\n";

        // True Vertices 
        {
            Eigen::MatrixXd vertices_true_mat = readMatrixFromFile(need("vertices_true.txt"));
            if (vertices_true_mat.rows() != N)
                throw std::runtime_error("Vertex count mismatch in vertices_true.txt");
            G.vertices_true.resize(N);
            for (int i = 0; i < N; ++i) G.vertices_true[i] = vertices_true_mat.row(i).transpose();
        }

        // True Vertices PUDQ 
        {
            Eigen::MatrixXd vertices_pudq_true_mat = readMatrixFromFile(need("vertices_pudq_true.txt"));
            if (vertices_pudq_true_mat.rows() != N)
                throw std::runtime_error("Vertex count mismatch in vertices_pudq_true.txt");
            G.vertices_pudq_true.resize(N);
            for (int i = 0; i < N; ++i) G.vertices_pudq_true[i] = vertices_pudq_true_mat.row(i).transpose();
        }

        // Edges 
        Eigen::MatrixXd edges_mat = readMatrixFromFile(need("edges.txt"));
        const int M = static_cast<int>(edges_mat.rows());
        if (M <= 0) throw std::runtime_error("No edges found in edges.txt");

        int min_id = INT_MAX, max_id = INT_MIN;
        for (int i = 0; i < M; ++i) {
            int a = static_cast<int>(edges_mat(i, 0));
            int b = static_cast<int>(edges_mat(i, 1));
            min_id = std::min(min_id, std::min(a, b));
            max_id = std::max(max_id, std::max(a, b));
        }

        enum class Indexing { Zero, One, Bad } idx = Indexing::Bad;
        if (min_id == 0 && max_id <= N - 1) idx = Indexing::Zero;
        else if (min_id >= 1 && max_id <= N) idx = Indexing::One;

        if (idx == Indexing::Bad) {
            throw std::runtime_error(
                "edges.txt has inconsistent ids (min=" + std::to_string(min_id) +
                ", max=" + std::to_string(max_id) + ", N=" + std::to_string(N) +
                "). Expected either 0..N-1 or 1..N.");
        }

        G.edges.resize(M);
        for (int i = 0; i < M; ++i) {
            int id1 = static_cast<int>(edges_mat(i, 0));
            int id2 = static_cast<int>(edges_mat(i, 1));
            if (idx == Indexing::One) { id1 -= 1; id2 -= 1; }
            if (id1 < 0 || id1 >= N || id2 < 0 || id2 >= N) {
                throw std::runtime_error("Edge " + std::to_string(i) +
                    " out of range after normalization: [" + std::to_string(id1) +
                    "," + std::to_string(id2) + "], N=" + std::to_string(N));
            }
            G.edges[i] = { id1, id2 };
        }
        std::cout << "Loaded " << M << " edges (from edges.txt, "
                << (idx == Indexing::Zero ? "0-based" : "1->0 normalized") << ")\n";

        // Delta Poses PUDQ 
        {
            Eigen::MatrixXd dp_pudq_mat = readMatrixFromFile(need("delta_poses_pudq.txt"));
            if (dp_pudq_mat.rows() != M)
                throw std::runtime_error("Edge count mismatch in delta_poses_pudq.txt");
            G.delta_poses_pudq.resize(M);
            G.delta_poses.resize(M);
            for (int i = 0; i < M; ++i) {
                G.delta_poses_pudq[i] = dp_pudq_mat.row(i).transpose();
                G.delta_poses[i]      = pudqlib::pudq_to_pose(G.delta_poses_pudq[i]);
            }
        }

        // Delta Poses PUDQ True 
        {
            Eigen::MatrixXd dp_pudq_true_mat = readMatrixFromFile(need("delta_poses_pudq_true.txt"));
            if (dp_pudq_true_mat.rows() != M)
                throw std::runtime_error("Edge count mismatch in delta_poses_pudq_true.txt");
            G.delta_poses_pudq_true.resize(M);
            G.delta_poses_true.resize(M);
            for (int i = 0; i < M; ++i) {
                G.delta_poses_pudq_true[i] = dp_pudq_true_mat.row(i).transpose();
                G.delta_poses_true[i]      = pudqlib::pudq_to_pose(G.delta_poses_pudq_true[i]);
            }
        }

        // t and R 
        {
            Eigen::MatrixXd t_mat = readMatrixFromFile(need("t.txt"));
            if (t_mat.rows() != M)
                throw std::runtime_error("Edge count mismatch in t.txt");
            G.t.resize(M);
            for (int i = 0; i < M; ++i) G.t[i] = t_mat.row(i).transpose();

            Eigen::MatrixXd R_mat = readMatrixFromFile(need("R.txt"));
            if (R_mat.rows() != M * 2 || R_mat.cols() != 2)
                throw std::runtime_error("R matrix size mismatch in R.txt (expect 2M x 2)");
            G.R.resize(M);
            for (int i = 0; i < M; ++i) G.R[i] = R_mat.block<2,2>(i * 2, 0);
        }

        // Information matrices 
        auto load_info3 = [&](const char* fname, std::vector<Eigen::Matrix3d>& out) {
            Eigen::MatrixXd mat = readMatrixFromFile(need(fname));
            if (mat.rows() != M * 3 || mat.cols() != 3)
                throw std::runtime_error(std::string(fname) + " size mismatch (expect 3M x 3)");
            out.resize(M);
            for (int i = 0; i < M; ++i) out[i] = mat.block<3,3>(i * 3, 0);
        };
        load_info3("information.txt",        G.information);
        load_info3("information_pudq.txt",  G.information_pudq);
        load_info3("information_se2.txt",   G.information_se2);

        auto load_vec1 = [&](const char* fname, std::vector<double>& out) {
            Eigen::MatrixXd v = readMatrixFromFile(need(fname));
            if (v.rows() != M) throw std::runtime_error(std::string("Edge count mismatch in ") + fname);
            out.resize(M);
            for (int i = 0; i < M; ++i) out[i] = v(i, 0);
        };
        load_vec1("kappa.txt", G.kappa);
        load_vec1("tau.txt",   G.tau);

        for (size_t e = 0; e < G.edges.size(); ++e) {
            auto [a,b] = G.edges[e];
            if (a < 0 || a >= N || b < 0 || b >= N) {
                throw std::runtime_error("Post-check: invalid edge " + std::to_string(e));
            }
        }

        std::cout << "Re-generating sparse Omega matrix...\n";
        G.Omega = pudqlib::Omega_sparse(G);

        std::cout << "Successfully loaded and reconstructed graph from MATLAB export.\n";
        return G;
    }

    void writeDoubleVectorToFile(const std::string& filename, const std::vector<double>& vec) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        file << std::fixed << std::setprecision(16);
        for (const double& val : vec) {
            file << val << "\n";
        }
    }
    
    void writeEdgesToFile(const std::string& filename, const std::vector<std::pair<int, int>>& edges) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        for (const auto& edge : edges) {
            file << edge.first << " " << edge.second << "\n";
        }
    }

    void saveFullAnalysisData(const std::string& output_dir, const pudqlib::Graph& optimized_graph, const std::vector<pudqlib::Robot>& robots) {
        
        namespace fs = std::filesystem;
        fs::path dir(output_dir);
        try {
            if (!fs::exists(dir)) {
                fs::create_directories(dir);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create directory: " + output_dir + " | " + e.what());
        }

        writeEigenVectorListToFile((dir / "optimized_vertices_pudq.txt").string(), optimized_graph.vertices_pudq);
        writeEigenVectorListToFile((dir / "optimized_vertices.txt").string(), optimized_graph.vertices);

        // Save GT 
        writeEigenVectorListToFile((dir / "true_vertices_pudq.txt").string(), optimized_graph.vertices_pudq_true);
        writeEigenVectorListToFile((dir / "true_vertices.txt").string(), optimized_graph.vertices_true);
        writeEigenVectorListToFile((dir / "true_delta_poses_pudq.txt").string(), optimized_graph.delta_poses_pudq_true);
        writeEigenVectorListToFile((dir / "true_delta_poses.txt").string(), optimized_graph.delta_poses_true);

        writeEdgesToFile((dir / "edges_0_indexed.txt").string(), optimized_graph.edges);

        // -Save Robot Data 
        for (const auto& robot : robots) {
            std::string r_str = std::to_string(robot.id);
            writeDoubleVectorToFile((dir / ("robot_" + r_str + "_cost.txt")).string(), robot.rgn_cost);
            writeDoubleVectorToFile((dir / ("robot_" + r_str + "_gradnorm.txt")).string(), robot.rgn_gradnorm);
        }
        
        std::cout << "Successfully exported all analysis data to " << output_dir << std::endl;
    }


    double wrap_theta(double theta) {
        if (theta > M_PI) {
            return theta - 2.0 * M_PI;
        } 
        else if (theta < -M_PI) {
            return theta + 2.0 * M_PI;
        } 
        else {
            return theta;
        }
    }

    double theta_diff(double theta_0, double theta_1) {
        const std::complex<double> I(0.0, 1.0); // sqrt(-1)
        std::complex<double> z = std::exp(theta_0 * I) * std::exp(-theta_1 * I);
        double dt = std::arg(z);
        return wrap_theta(dt);
    }

    void reorient_graph_to_identity(pudqlib::Graph& G) {
        if (G.vertices_pudq.empty()) return;

        pudqlib::PUDQ x_0 = G.vertices_pudq[0];
        pudqlib::PUDQ x_0_inv = pudqlib::pudq_inv(x_0);

        for (size_t i = 0; i < G.vertices_pudq.size(); ++i) {
            G.vertices_pudq[i] = pudqlib::pudq_mul(x_0_inv, G.vertices_pudq[i]);
            G.vertices[i] = pudqlib::pudq_to_pose(G.vertices_pudq[i]);
        }
        std::cout << "  -> Graph reoriented to Identity (0,0,0)." << std::endl;
    }
    
    void align_graph_to_ground_truth(pudqlib::Graph& G) {
        if (G.vertices.empty()) return;
        
        pudqlib::PUDQ x_0_opt = G.vertices_pudq[0];
        pudqlib::PUDQ x_0_gt = G.vertices_pudq_true[0];
        pudqlib::PUDQ x_align = pudqlib::pudq_mul(x_0_gt, pudqlib::pudq_inv(x_0_opt));

        for (size_t i = 0; i < G.vertices_pudq.size(); ++i) {
            G.vertices_pudq[i] = pudqlib::pudq_mul(x_align, G.vertices_pudq[i]);
            G.vertices[i] = pudqlib::pudq_to_pose(G.vertices_pudq[i]);
        }
        std::cout << "  -> Graph aligned to Ground Truth anchor." << std::endl;
    }


    // ATE_eucl
    void calculate_ATE_Euclidean(const pudqlib::Graph& G) {
        double ATE = 0.0;
        double linear_error = 0.0;
        double angular_error = 0.0;
        int num_vertices = static_cast<int>(G.vertices.size());

        for (int i = 0; i < num_vertices; ++i) {
            Eigen::Vector3d x_i_true = G.vertices_true[i];
            Eigen::Vector3d x_i_hat = G.vertices[i];

            Eigen::Vector3d eij = Eigen::Vector3d::Zero();
            eij.head<2>() = x_i_true.head<2>() - x_i_hat.head<2>();
            eij(2) = theta_diff(x_i_true(2), x_i_hat(2));

            linear_error += eij.head<2>().dot(eij.head<2>());
            angular_error += eij(2) * eij(2);
            ATE += eij.dot(eij);
        }

        double linear = std::sqrt(linear_error / num_vertices);
        double angular = std::sqrt(angular_error / num_vertices);
        ATE = std::sqrt(ATE / num_vertices);

        std::printf("ATE (Euclidean): %.6f (Linear: %.6f, Angular: %.6f)\n", ATE, linear, angular);
    }

    // RPE_eucl
    void calculate_RPE_Euclidean(const pudqlib::Graph& G) {
        double RPE = 0.0;
        double linear_error = 0.0;
        double angular_error = 0.0;
        int num_edges = static_cast<int>(G.edges.size());

        for (int ij = 0; ij < num_edges; ++ij) {
            Eigen::Vector3d z_ij_true = G.delta_poses_true[ij];

            int idx_i = G.edges[ij].first;
            int idx_j = G.edges[ij].second;
            Eigen::Vector3d x_i = G.vertices[idx_i];
            Eigen::Vector3d x_j = G.vertices[idx_j];

            Eigen::Matrix2d R_i = pudqlib::R_from_theta(x_i(2));

            Eigen::Vector2d dt = R_i.transpose() * (x_j.head<2>() - x_i.head<2>());
            double dth = wrap_theta(x_j(2) - x_i(2));
            Eigen::Vector3d z_ij_hat;
            z_ij_hat << dt(0), dt(1), dth;

            Eigen::Vector3d eij = Eigen::Vector3d::Zero();
            eij.head<2>() = z_ij_true.head<2>() - z_ij_hat.head<2>();
            eij(2) = theta_diff(z_ij_true(2), z_ij_hat(2));

            linear_error += eij.head<2>().dot(eij.head<2>());
            angular_error += eij(2) * eij(2);
            RPE += eij.dot(eij);
        }

        double linear = std::sqrt(linear_error / num_edges);
        double angular = std::sqrt(angular_error / num_edges);
        RPE = std::sqrt(RPE / num_edges);

        std::printf("RPE (Euclidean): %.6f (Linear: %.6f, Angular: %.6f)\n", RPE, linear, angular);
    }

    // ATE_pudq
    void calculate_ATE_PUDQ(const pudqlib::Graph& G) {
        double ATE = 0.0;
        int num_vertices = static_cast<int>(G.vertices.size());

        pudqlib::PUDQ id; id << 1.0, 0.0, 0.0, 0.0; 

        for (int i = 0; i < num_vertices; ++i) {
            pudqlib::PUDQ x_i_true = G.vertices_pudq_true[i];
            pudqlib::PUDQ x_i_hat = G.vertices_pudq[i];

            pudqlib::PUDQ err_q = pudqlib::pudq_compose(pudqlib::pudq_inv(x_i_true), x_i_hat);
            Eigen::Vector3d eij = pudqlib::Log_1(err_q);

            ATE += eij.dot(eij);
        }

        ATE = std::sqrt(ATE / num_vertices);
        std::printf("ATE (PUDQ):      %.6f\n", ATE);
    }

    // RPE_pudq
    void calculate_RPE_PUDQ(const pudqlib::Graph& G) {
        double RPE = 0.0;
        int num_edges = static_cast<int>(G.edges.size());

        for (int k = 0; k < num_edges; ++k) {
            pudqlib::PUDQ z_ij_true = G.delta_poses_pudq_true[k];
            
            int i = G.edges[k].first;
            int j = G.edges[k].second;
            pudqlib::PUDQ x_i = G.vertices_pudq[i];
            pudqlib::PUDQ x_j = G.vertices_pudq[j];

            pudqlib::PUDQ z_est = pudqlib::pudq_compose(pudqlib::pudq_inv(x_i), x_j);
            pudqlib::PUDQ err_q = pudqlib::pudq_compose(pudqlib::pudq_inv(z_ij_true), z_est);
            
            Eigen::Vector3d eij = pudqlib::Log_1(err_q);

            RPE += eij.dot(eij);
        }

        RPE = std::sqrt(RPE / num_edges);
        std::printf("RPE (PUDQ):      %.6f\n", RPE);
    }


}