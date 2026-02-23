### Running the Simulation

1. Run `Mat_to_C.m` to convert the MATLAB data into a `.g2o` file readable by C++.
2. Create a Catkin workspace and place all source files (excluding the MATLAB script) into the `src` folder.
3. Build the project using `catkin_make`.
4. Source the environment: `source devel/setup.bash`.
5. Launch the simulation: `roslaunch distributed_pgo start_sim.launch mode:=2`. 
   * *Mode 1: Truly distributed stop method.*
   * *Mode 2: Forced synchronized stop.*
6. Evaluate the results: `./devel/lib/distributed_pgo/evaluate_results 5 /path/to/your/export_folder/`.
