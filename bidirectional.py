import time
import cv2
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

def obstacle_map(canvas):
    # Creating boundary
    cv2.rectangle(canvas, (0, 0), (width, height), (0, 0, 255), -1)
    cv2.rectangle(canvas, (0+total_clearance, 0+total_clearance), (width-total_clearance, height-total_clearance), (255, 255, 255), -1)

    # Creating rectangle 1
    cv2.rectangle(canvas, pt1=(1750//scaling_factor + total_clearance,1000//scaling_factor -total_clearance),pt2=(1500//scaling_factor - total_clearance,2000//scaling_factor + total_clearance), color=(0,0,255), thickness=-1)
    cv2.rectangle(canvas,pt1=(1750//scaling_factor ,1000//scaling_factor),pt2=(1500//scaling_factor ,2000//scaling_factor),color=(179,41,43),thickness=-1)

    # Creating rectangle 2
    cv2.rectangle(canvas,pt1=(2500//scaling_factor - total_clearance,1000//scaling_factor + total_clearance),pt2=(2750//scaling_factor + total_clearance,0 - total_clearance),color=(0,0,255),thickness=-1)
    cv2.rectangle(canvas,pt1=(2500//scaling_factor, 1000//scaling_factor),pt2=(2750//scaling_factor ,0),color=(179,41,43),thickness=-1)
    
    # Creating a Circle with the clearance
    center = (4200//scaling_factor, 1200//scaling_factor)
    radius = 600//scaling_factor + total_clearance
    cv2.circle(canvas, center, radius, (0, 0, 255), -1)

    # Draw the original circle
    radius = 600//scaling_factor
    cv2.circle(canvas, center, radius, (179, 41, 43), -1)
    return canvas

def input_coordinates():
    while True:
        start_node_str = input("Enter the coordinates of starting node (x,y,theta):")
        goal_node_str = input("Enter the coordinates of Goal node (x,y):")
        
        start_node = tuple(map(int, start_node_str.split(',')))
        goal_node = tuple(map(int, goal_node_str.split(',')))

        start_node = (start_node[0] // scaling_factor, start_node[1] // scaling_factor, start_node[2])
        goal_node = (goal_node[0] // scaling_factor, goal_node[1] // scaling_factor, goal_node[2])
        

        # Check if the start and goal node are valid
        if is_valid(start_node[0],start_node[1]):
            if is_valid(goal_node[0],goal_node[1]):
                break
            else:
                print("Invalid goal node. Please enter valid coordinates.")
                continue
        else:
            print("Invalid start node. Please enter valid coordinates.")
            continue

    return start_node,goal_node

def input_clearance(key):
    while True:
        try:
            radius = int(input("Enter the " + key + " clearance (in mm): "))
            if radius < 0:
                print(key +" clearance cannot be negative. Please enter a non-negative value.")
            else:
                break           
        except ValueError:
            print("Invalid input. Please enter a valid integer for the " + key + " clearance.")

    return radius

def input_rpm():
    rpm_str = input("Enter the RPMs for the wheels [RPM1,RPM2]:")
    rpm = tuple(map(int, rpm_str.split(',')))

    return rpm

def is_valid(x,y):
    # Check if the coordinates are in bounds of canvas
    if (0 + total_clearance <= x <= width - total_clearance and 0+ total_clearance <= y <= height-total_clearance):
        pass
    else:
        return False

    # Check if the coordinates are within obstacle region
    if np.array_equal(canvas[int(y), int(x)], [179, 41, 43]) or np.array_equal(canvas[int(y), int(x)], [0,0,255]):
        return False
 
    return True

def rpm2cord(actions,present_node):
    # Robot parameters
    r = 0.033*100 # wheel radius in cm 
    L = 0.287*100 # distance between wheel in cm

    # Wheel velocity
    left_v = round(((2 * np.pi * actions[0]) / 60),1)
    right_v = round(((2 * np.pi * actions[1]) / 60),1)

    # Initializing Variables
    d_theta = math.radians(present_node[2])
    d_x = present_node[0]
    d_y = present_node[1]
    
    dt = 1
    t = 0
    t_step = 1

    # Distance travelled
    action_cost = 0

    while t < t_step:
        t = t + dt
        d_theta = round((d_theta+(r / L) * (right_v - left_v) * dt),1)
        d_x = d_x + 0.5*r * (left_v + right_v) * math.cos(d_theta) * dt
        d_y = d_y + 0.5*r * (left_v + right_v) * math.sin(d_theta) * dt

        action_cost = round((action_cost + math.sqrt(math.pow(d_x,2)+math.pow(d_y,2))),1)

        if is_valid(d_x,d_y):
            pass
        else:
            return (None,None)
        
    # Keeping theta in 0 to 2*pi range
    angle = d_theta % (2 * math.pi)
    angle = round(math.degrees(angle))

    # x & y in cm
    x = round(d_x,2)
    y = round(d_y,2)

    return (d_x, d_y, angle), action_cost

def action(present_node):
    movements=[]
    action_list = [(0,rpm1),(rpm1,0),(rpm1,rpm1),(0,rpm2),(rpm2,0),(rpm2,rpm2),(rpm1,rpm2),(rpm2,rpm1)]

    # 8 actions
    for actions in action_list:
        next_node,action_cost = rpm2cord(actions,present_node)

        if next_node or action_cost is not None:
            movements.append((next_node,action_cost))
        else:
            movements.append((None,None))

    return movements

def heuristic_cost(current_pos,goal_pos):
    cost = math.sqrt((goal_pos[1]-current_pos[1])**2+(goal_pos[0]-current_pos[0])**2) 
    return round(cost,1)

def round_node(node):
    x = round(node[0] * 2) / 2
    y = round(node[1] * 2) / 2
    theta = node[2]

    return x,y,theta

def get_path(start_position, goal_position, closed_list_forward, closed_list_backward):
    path_forward = []
    path_backward = []
    current_node = goal_position

    # Backtrack from goal node to intersection node
    while current_node != start_position:
        path_backward.append(current_node)
        current_node = tuple(closed_list_backward[current_node])

    current_node = start_position
    # Backtrack from start node to intersection node
    while current_node != goal_position:
        path_forward.append(current_node)
        current_node = tuple(closed_list_forward[current_node])

    # Combine the two partial paths
    path = path_forward + path_backward[::-1]

    return path

def visualization(path, canvas, start_position, goal_position, frame_skip=450):
    # Canvas cm to mm
    resized_canvas = cv2.resize(canvas, (canvas.shape[1]*scaling_factor, canvas.shape[0]*scaling_factor))

    output_video = cv2.VideoWriter('astar.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 50, (resized_canvas.shape[1], resized_canvas.shape[0]))
    skip_counter = 0

    # Draw start and goal node
    cv2.circle(resized_canvas, (int(start_position[0]*scaling_factor), int(start_position[1]*scaling_factor)), 5, (0, 255, 0), -1)
    cv2.circle(resized_canvas, (int(goal_position[0]*scaling_factor), int(goal_position[1]*scaling_factor)), 5, (0, 0, 255), -1)

    # for key, values in closed_list.items():
    #     if values is None:
    #         continue
        
    #     # Child
    #     pt1 = (int(key[0]*scaling_factor), int(key[1]*scaling_factor))
    #     # Parent
    #     pt2 = (int(values[0]*scaling_factor), int(values[1]*scaling_factor))

    #     cv2.arrowedLine(resized_canvas, pt2, pt1, (255,255,0), 1)

    #     skip_counter += 1
    #     if skip_counter == frame_skip:
    #         vid = cv2.flip(resized_canvas, 0)
    #         output_video.write(vid)
    #         skip_counter = 0

    # Draw the full path on the canvas
    for i in range(len(path) - 1):
        cv2.arrowedLine(resized_canvas, (int(path[i][0]*scaling_factor), int(path[i][1]*scaling_factor)), (int(path[i+1][0]*scaling_factor), int(path[i+1][1]*scaling_factor)), (0, 0, 0), thickness=2)
        vid = cv2.flip(resized_canvas, 0) 
        output_video.write(vid)

    output_video.release()


def bidirectional_a_star(start_position, goal_position, canvas, goal_threshold_distance=1.5):
    # List of nodes to be explored
    open_list_forward = []
    open_list_backward = []
    
    # Dictionary stores explored and its parent node
    closed_list_forward = {}
    closed_list_backward = {}

    # Dictionary to store node information {present_node: [parent_node, cost_to_come]}
    node_info_forward = {}
    node_info_backward = {}

    # Closed set stores nodes that are visited (as nearest 0.5 multiple)
    closed_set_forward = set()
    closed_set_backward = set()

    # Visited nodes (as nearest 0.5 multiple)
    visited_nodes_forward = np.zeros(((height*2)+1,(width+1)*2), dtype=int)
    visited_nodes_backward = np.zeros(((height*2)+1,(width+1)*2), dtype=int)

    # Heap to store the nodes based on their cost value
    hq.heapify(open_list_forward)
    hq.heapify(open_list_backward)

    # Inserting the initial node with its [total_cost, present_node]
    hq.heappush(open_list_forward, [ 0+heuristic_cost(start_position,goal_position), start_position])
    hq.heappush(open_list_backward, [ 0+heuristic_cost(goal_position,start_position), goal_position])

    # Set the node_info for the start and goal positions
    node_info_forward[start_position] = [None, 0]
    node_info_backward[goal_position] = [None, 0]

    # Visited set for start and goal positions
    index = round_node(start_position)
    visited_nodes_forward[int(index[1]*2)][int(index[0]*2)] = 1

    index = round_node(goal_position)
    visited_nodes_backward[int(index[1]*2)][int(index[0]*2)] = 1

    while open_list_forward and open_list_backward:
        # Forward search
        total_cost, present_node = hq.heappop(open_list_forward)
        parent_node, cost2come = node_info_forward[present_node]
        closed_list_forward[present_node] = parent_node
        closed_set_forward.add(round_node(present_node))

        # Backward search
        total_cost, present_node = hq.heappop(open_list_backward)
        parent_node, cost2come = node_info_backward[present_node]
        closed_list_backward[present_node] = parent_node
        closed_set_backward.add(round_node(present_node))

        # Check if the two searches have intersected
        if present_node in closed_set_forward or present_node in closed_set_backward:
            intersection_node = present_node
            return get_path(start_position, goal_position, closed_list_forward, closed_list_backward)

        # Add neighboring nodes to the open lists
        for next_node,action_cost in action(present_node):
            if next_node is not None:
                rounded_next_node = round_node(next_node)

                # Calculating the index of visited nodes array
                scaled_x = int(rounded_next_node[1] * 2)
                scaled_y = int(rounded_next_node[0] * 2)

                # Forward search
                if (rounded_next_node not in closed_set_forward):
                    if (visited_nodes_forward[scaled_x,scaled_y] == 0):
                        new_cost2come = cost2come + action_cost
                        new_total_cost = new_cost2come + heuristic_cost(next_node, goal_position)

                        hq.heappush(open_list_forward, [new_total_cost, next_node])
                        node_info_forward[next_node] = [present_node, new_cost2come]

                        # Set visited node as 1
                        visited_nodes_forward[scaled_x,scaled_y] = 1
                    
                    # If node is already in open list, we need to compare cost and update if needed
                    else:
                        if (next_node in node_info_forward) and (cost2come + action_cost) < node_info_forward[next_node][1]:
                            new_cost2come = cost2come + action_cost
                            new_total_cost = new_cost2come + heuristic_cost(next_node, goal_position)
                            hq.heappush(open_list_forward, [new_total_cost, next_node])
                            node_info_forward[next_node] = [present_node, new_cost2come]

                # Backward search
                if (rounded_next_node not in closed_set_backward):
                    if (visited_nodes_backward[scaled_x,scaled_y] == 0):
                        new_cost2come = cost2come + action_cost
                        new_total_cost = new_cost2come + heuristic_cost(next_node, start_position)

                        hq.heappush(open_list_backward, [new_total_cost, next_node])
                        node_info_backward[next_node] = [present_node, new_cost2come]

                        # Set visited node as 1
                        visited_nodes_backward[scaled_x,scaled_y] = 1
                    
                    # If node is already in open list, we need to compare cost and update if needed
                    else:
                        if (next_node in node_info_backward) and (cost2come + action_cost) < node_info_backward[next_node][1]:
                            new_cost2come = cost2come + action_cost
                            new_total_cost = new_cost2come + heuristic_cost(next_node, start_position)
                            hq.heappush(open_list_backward, [new_total_cost, next_node])
                            node_info_backward[next_node] = [present_node, new_cost2come]

    return "Solution does not exist"

if __name__=="__main__":
    start_time = time.time() 

    # Scaling factor
    scaling_factor = 10

    # Create blank canvas (mm to cm)
    width = 6000//scaling_factor
    height = 2000//scaling_factor
    canvas = np.ones((height,width,3), dtype=np.uint8) * 255

    # Input clearance and robot radius
    robot_radius = 220
    robot_clearance = input_clearance("robot")
    obstacle_clearance = input_clearance("obstacle")

    total_clearance =  (robot_clearance + obstacle_clearance + robot_radius)//scaling_factor

    # Draw the obstacle map
    canvas = obstacle_map(canvas)

    # Input start and goal node coordinates
    start_position,goal_position = input_coordinates()

    # Input RPMs
    rpm1,rpm2 = input_rpm()

    # Bidirectional A*
    path = bidirectional_a_star(start_position, goal_position, canvas)

    # Goal reached
    goal_reached_time = time.time()
    print("Total time taken to reach the goal: ",goal_reached_time-start_time)

    # Display Node exploration and Optimal path
    visualization(path,canvas,start_position,goal_position)

    end_time = time.time()

    print("Total time taken to execute the code: ",end_time-start_time)


    