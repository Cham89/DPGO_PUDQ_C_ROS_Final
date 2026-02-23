Step 1: Run Mat_to_C.m to generate the g2o file into an readible input for C++ 
Step 2: create a workspace and put rest of the files (except Mat_to_C.m) into the workspace 
Step 3: catkin_make
Step 4: source devel/setup.bash
Step 5: roslaunch distributed_pgo start_sim.launch mode:=2   
			( mode 1 means truely distributed stop method / mode 2 means if one robot stops it forces other all to stop together)
Step 6: ./devel/lib/distributed_pgo/evaluate_results 5 /home/hsuanpin/DPGO_PUDQ_Matlab/	DPGO_PUDQ_Matlab/matlab_export_c
	
