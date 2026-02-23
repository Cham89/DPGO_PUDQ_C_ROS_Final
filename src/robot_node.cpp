#include "pudq_lib.hpp"
#include "graph.hpp"
#include "optimization.hpp"

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include "distributed_pgo/PgoUpdate.h"
#include "distributed_pgo/PgoStatus.h"

#include <thread>
#include <mutex>
#include <random>
#include <vector>
#include <filesystem>

class RobotNode {
public:
    RobotNode(ros::NodeHandle& nh) : m_nh(nh), m_gen(m_rd()) {
        m_nh.param<int>("robot_id", m_robot_id, 0);
        m_nh.param<std::string>("graph_directory", m_graph_dir, "");
        m_nh.param<int>("num_robots", m_num_robots, 1);
        m_nh.param<double>("mu_reg", m_mu_reg, 1e-1);
        m_nh.param<double>("grad_tol", m_grad_tol, 1e-2);
        m_nh.param<double>("poisson_lambda", m_lambda, 1.0); 
        m_nh.param<double>("comm_delay", m_comm_delay, 0.2);

        m_nh.param<int>("shutdown_mode", m_shutdown_mode, 1);

        m_exp_dist = std::exponential_distribution<double>(m_lambda);

        m_robots_converged.resize(m_num_robots, false);

        ROS_INFO("Robot %d: Loading graph...", m_robot_id);
        pudqlib::Graph full_graph = graph::load_graph_from_matlab_export(m_graph_dir);
        std::vector<pudqlib::MultiGraph> multi_graphs = graph::distributebiggraph_withinverseedge(full_graph, m_num_robots);
        m_local_graph = graph::build_local_subgraph(multi_graphs, m_robot_id);

        m_update_pub = m_nh.advertise<distributed_pgo::PgoUpdate>("/pgo_updates", 100);
        m_update_sub = m_nh.subscribe("/pgo_updates", 1000, &RobotNode::communicationCallback, this);
        
        m_status_pub = m_nh.advertise<distributed_pgo::PgoStatus>("/pgo_status", 100);
        m_status_sub = m_nh.subscribe("/pgo_status", 100, &RobotNode::statusCallback, this);

        m_global_signal_pub = m_nh.advertise<std_msgs::Bool>("/pgo_global_signal", 10, true);
        m_global_signal_sub = m_nh.subscribe("/pgo_global_signal", 100, &RobotNode::globalSignalCallback, this);
    }

    // ======================================================================
    // Optimization
    // ======================================================================
    void runOptimizationLoop() {
        while (ros::ok()) {
            if (m_global_convergence_reached) {
                ros::Duration(0.5).sleep();
                continue;
            }

            ros::Duration(m_exp_dist(m_gen)).sleep();

            pudqlib::LocalGraph graph_copy;
            {
                std::lock_guard<std::mutex> lock(m_graph_mutex);
                graph_copy = m_local_graph; 
            }

            auto [F_now, norm_grad_F] = opt::Optimization_RLM_inverse_edge(graph_copy, m_mu_reg, m_grad_tol);

            {
                std::lock_guard<std::mutex> lock(m_graph_mutex);
                m_local_graph = graph_copy; 
            }

            bool am_i_converged = (norm_grad_F < m_grad_tol);
            {
                std::lock_guard<std::mutex> lock(m_status_mutex);
                m_robots_converged[m_robot_id] = am_i_converged;
            }

            distributed_pgo::PgoStatus status_msg;
            status_msg.sender_robot_id = m_robot_id;
            status_msg.is_converged = am_i_converged;
            status_msg.current_grad_norm = norm_grad_F;
            m_status_pub.publish(status_msg);
            
            checkGlobalConvergence();

            ROS_INFO("Robot %d: Grad: %.4f", m_robot_id, norm_grad_F);
        }
    }

    // ======================================================================
    // Communication Send 
    // ======================================================================
    void runCommunicationSendLoop() {
        if (m_comm_delay <= 0) return;
        ros::Rate comm_rate(1.0 / m_comm_delay);

        while (ros::ok()) {
            if (m_global_convergence_reached) break;

            comm_rate.sleep();
            std::vector<pudqlib::PUDQ> vertices_pudq_copy;
            std::vector<Eigen::Vector3d> vertices_vec3_copy;
            {
                std::lock_guard<std::mutex> lock(m_graph_mutex);
                vertices_pudq_copy = m_local_graph.vertices_pudq;
                vertices_vec3_copy = m_local_graph.vertices;
            } 

            for (int i = 0; i < vertices_pudq_copy.size(); ++i) {
                publishVertexUpdate(i, vertices_pudq_copy[i], vertices_vec3_copy[i]);
            }
        }
    }

    // ======================================================================
    // Communication Receive 
    // ======================================================================
    void communicationCallback(const distributed_pgo::PgoUpdate::ConstPtr& msg) {
        if (msg->sender_robot_id == m_robot_id) return;

        bool graph_updated = false;
        double current_grad_norm = 999.9;

        {
            std::lock_guard<std::mutex> lock(m_graph_mutex);

            for (int L = 0; L < m_local_graph.lm_foreign_info.size(); ++L) {
                const auto& info = m_local_graph.lm_foreign_info[L];
                if (info.robot == msg->sender_robot_id && info.local_index == msg->local_vertex_index) {
                    
                    m_local_graph.lm_vertices_pudq[L](0) = msg->pose_pudq[0];
                    m_local_graph.lm_vertices_pudq[L](1) = msg->pose_pudq[1];
                    m_local_graph.lm_vertices_pudq[L](2) = msg->pose_pudq[2];
                    m_local_graph.lm_vertices_pudq[L](3) = msg->pose_pudq[3];
                    m_local_graph.lm_vertices[L](0) = msg->pose_vec3[0];
                    m_local_graph.lm_vertices[L](1) = msg->pose_vec3[1];
                    m_local_graph.lm_vertices[L](2) = msg->pose_vec3[2];
                    
                    graph_updated = true;
                    break;
                }
            }

            if (graph_updated) {
                pudqlib::Result result = pudqlib::rgn_gradhess_multi_J_forinverse(m_local_graph);
                
                const bool anchor_first = m_local_graph.anchor_first && (m_local_graph.robot_id == 0);
                const int num_vertices = static_cast<int>(m_local_graph.vertices.size());
                const int total_dim = 4 * num_vertices;
                const int offset = anchor_first ? 4 : 0;
                const int dim = total_dim - offset;

                current_grad_norm = result.rgrad_X.segment(offset, dim).norm();

            }
        } 

        if (graph_updated) {
            bool am_i_converged = (current_grad_norm < m_grad_tol);

            {
                std::lock_guard<std::mutex> stat_lock(m_status_mutex);
                m_robots_converged[m_robot_id] = am_i_converged;
            }

            distributed_pgo::PgoStatus status_msg;
            status_msg.sender_robot_id = m_robot_id;
            status_msg.is_converged = am_i_converged;
            status_msg.current_grad_norm = current_grad_norm;
            m_status_pub.publish(status_msg);
            
        }
    }

    // ======================================================================
    // Status Receive 
    // ======================================================================
    void statusCallback(const distributed_pgo::PgoStatus::ConstPtr& msg) {
        if (msg->sender_robot_id == m_robot_id) return;

        {
            std::lock_guard<std::mutex> lock(m_status_mutex);
            if (msg->sender_robot_id < m_num_robots) {
                m_robots_converged[msg->sender_robot_id] = msg->is_converged;
            }
        } 

        checkGlobalConvergence();
    }

    // ======================================================================
    // Global Signal Receive 
    // ======================================================================
    void globalSignalCallback(const std_msgs::Bool::ConstPtr& msg) {
        if (msg->data == true && m_shutdown_mode == 2 && !m_global_convergence_reached) {
            m_global_convergence_reached = true;
            ROS_WARN("Robot %d: RECEIVED FORCE SHUTDOWN SIGNAL!", m_robot_id);
            ros::shutdown();
        }
    }

    // ======================================================================
    // Save & Print
    // ======================================================================
    void saveFinalResults() {
        pudqlib::Result result_opt;
        {
            std::lock_guard<std::mutex> lock(m_graph_mutex);
            result_opt = pudqlib::rgn_gradhess_multi_J_forinverse(m_local_graph);
        }
        
        const bool anchor_first = m_local_graph.anchor_first && (m_local_graph.robot_id == 0);
        const int num_vertices = static_cast<int>(m_local_graph.vertices.size());
        const int total_dim = 4 * num_vertices;
        const int offset = anchor_first ? 4 : 0;
        const int dim = total_dim - offset;

        double final_cost = result_opt.F_X;
        double final_grad = result_opt.rgrad_X.segment(offset, dim).norm();

        std::printf("\n*** Robot %d Finished! ***\n", m_robot_id);
        std::printf("Final Local Cost=%.6g, Final Local Grad=%.6g\n", final_cost, final_grad);

        try {
            namespace fs = std::filesystem;
            std::string output_dir = "cpp_export_ros/robot_" + std::to_string(m_robot_id);
            
            if (!fs::exists(output_dir)) {
                fs::create_directories(output_dir);
            }
        
            graph::writeEigenVectorListToFile(output_dir + "/vertices_pudq.txt", m_local_graph.vertices_pudq);
            graph::writeEigenVectorListToFile(output_dir + "/vertices_vec3.txt", m_local_graph.vertices);
            
            std::cout << "Exported local graph to: " << output_dir << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error during export: " << e.what() << std::endl;
        }
    }

private:
    void publishVertexUpdate(int local_idx, const pudqlib::PUDQ& pudq, const Eigen::Vector3d& vec3) {
        distributed_pgo::PgoUpdate msg;
        msg.sender_robot_id = m_robot_id;
        msg.local_vertex_index = local_idx;
        msg.pose_pudq[0] = pudq(0); msg.pose_pudq[1] = pudq(1);
        msg.pose_pudq[2] = pudq(2); msg.pose_pudq[3] = pudq(3);
        msg.pose_vec3[0] = vec3(0); msg.pose_vec3[1] = vec3(1); msg.pose_vec3[2] = vec3(2);
        m_update_pub.publish(msg);
    }

    void checkGlobalConvergence() {
        bool all_true = true;
        {
            std::lock_guard<std::mutex> lock(m_status_mutex);
            for (bool c : m_robots_converged) {
                if (!c) {
                    all_true = false;
                    break;
                }
            }
        }

        if (all_true && !m_global_convergence_reached) {
            m_global_convergence_reached = true;

            if (m_shutdown_mode == 2) {
                std_msgs::Bool force_msg;
                force_msg.data = true;
                m_global_signal_pub.publish(force_msg);

                ros::Duration(0.1).sleep(); 
            }

            ROS_WARN("***************************************");
            ROS_WARN("ROBOT %d DETECTED GLOBAL CONVERGENCE!", m_robot_id);
            ROS_WARN("***************************************");
            ros::shutdown();
        }
    }

    int m_robot_id;
    int m_num_robots;
    std::string m_graph_dir;

    int m_shutdown_mode;
    
    pudqlib::LocalGraph m_local_graph;
    std::mutex m_graph_mutex; 

    std::vector<bool> m_robots_converged; 
    std::mutex m_status_mutex; 
    bool m_global_convergence_reached = false;

    double m_lambda, m_mu_reg, m_grad_tol, m_comm_delay;
    std::random_device m_rd;
    std::mt19937 m_gen;
    std::exponential_distribution<double> m_exp_dist;

    ros::NodeHandle m_nh;
    ros::Subscriber m_update_sub;
    ros::Subscriber m_status_sub; 
    ros::Publisher m_update_pub;
    ros::Publisher m_status_pub;  

    ros::Publisher m_global_signal_pub;
    ros::Subscriber m_global_signal_sub;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "robot_pgo_node");
    ros::NodeHandle nh("~");
    RobotNode robot(nh);
    std::thread optimization_thread(&RobotNode::runOptimizationLoop, &robot);
    std::thread comm_send_thread(&RobotNode::runCommunicationSendLoop, &robot);
    ros::spin();
    optimization_thread.join();
    comm_send_thread.join();
    std::cout << "\nNode stopped. Saving final results..." << std::endl;
    robot.saveFinalResults();
    return 0;
}