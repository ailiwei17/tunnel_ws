source ~/tunnel_ws/devel/setup.bash
{
gnome-terminal -x bash -c "roslaunch mapping_mid_360.launch;exec bash"
}&
sleep 3s
{
gnome-terminal -x bash -c "roslaunch map_generate.launch;exec bash"
}



