# One 3D detection. x/y/z are ENU metres; drone_* is the ENU drone position.
string class_name
float32 confidence
float64 stamp
float32 x
float32 y
float32 z
float32 drone_x
float32 drone_y
float32 drone_z
string source          # rgb | thermal | fused
int32 bbox_x
int32 bbox_y
int32 bbox_w
int32 bbox_h
string image_path
string method          # depth | ground_plane
bool is_new            # False when suppressed as a repeat of the same target
