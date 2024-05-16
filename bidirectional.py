import numpy as np
import cv2
import math
from queue import PriorityQueue as PQueue
import time
import matplotlib.pyplot as plt

# Initialize variables
map_width = 600
map_height = 300

start_coords = (0, 150)
end_coords = (599, 150)
start_orientation = 0
end_orientation = 180
clearance_value = 10
robot_size = 5
weight_factor = 1
step_length = 5
goal_node = None
goal_reached = False

# Create the map
map_data = np.zeros((map_height, map_width), dtype=int)
normal_map = np.zeros((map_height, map_width), dtype=int)
map_image = np.full((map_height, map_width, 3), 255, dtype=np.uint8)
# print("Creating Map with obstacles...")

for y in range(map_height):
    for x in range(map_width):
        if (y - 90) ** 2 + (x - 263) ** 2 <= 70 ** 2 or (y - 220) ** 2 + (x - 445) ** 2 <= 37.5 ** 2 or (
                y - 242.5) ** 2 + (x - 112) ** 2 <= 40 ** 2:
            map_data[y, x] = 1

# print("Map Created")

for y in range(map_height):
    for x in range(map_width):
        if map_data[y, x] == 1:
            map_image[map_height - y - 1, x] = (255, 90, 90)

for y in range(map_height):
    for x in range(map_width):
        if (y - 90) ** 2 + (x - 263) ** 2 <= 70 ** 2 or (y - 220) ** 2 + (x - 445) ** 2 <= 37.5 ** 2 or (
                y - 242.5) ** 2 + (x - 112) ** 2 <= 40 ** 2:
            normal_map[y, x] = 1

for y in range(map_height):
    for x in range(map_width):
        if normal_map[y, x] == 1:
            map_image[map_height - y - 1, x] = (0, 0, 255)

# Initialize queues, dictionaries, and lists
forward_queue = PQueue()
forward_queue.put((0, (start_coords, start_orientation)))
forward_edge_cost = {start_coords: 0}
forward_total_cost = {start_coords: 0}
forward_parent = {start_coords: start_coords}
forward_visited = [[[0 for _ in range(12)] for _ in range(map_width)] for _ in range(map_height)]
forward_visited_list = []
forward_angle_dict = {}

backward_queue = PQueue()
backward_queue.put((0, (end_coords, end_orientation)))
backward_edge_cost = {end_coords: 0}
backward_total_cost = {end_coords: 0}
backward_parent = {end_coords: end_coords}
backward_visited = [[[0 for _ in range(12)] for _ in range(map_width)] for _ in range(map_height)]
backward_visited_list = []
backward_angle_dict = {}

goal_threshold = 2 * (step_length + robot_size)

def mark_visited(node, direction):
    x = node[0]
    y = node[1]
    theta = node[2]
    orientation = int(((360 + theta) % 360) / 30)
    if direction == 'forward':
        forward_visited[y][x][orientation] = True
    else:
        backward_visited[y][x][orientation] = True

def check_visited(node, direction):
    x = node[0]
    y = node[1]
    theta = node[2]
    orientation = int(((360 + theta) % 360) / 30)

    if direction == 'forward':
        visited_matrix = forward_visited
    else:
        visited_matrix = backward_visited

    for flag in visited_matrix[y][x]:
        if flag:
            return True
    return False

def convert_coords(x, y):
    y_cv2 = (map_height - 1) - y
    x_cv2 = x
    return x_cv2, y_cv2

def move_robot(node, angle_change):
    new_theta = (node[1] + angle_change) % 360
    updated_x = round(node[0][0] + step_length * math.cos(math.radians(new_theta)), 1)
    updated_y = round(node[0][1] + step_length * math.sin(math.radians(new_theta)), 1)
    return int(updated_x), int(updated_y), new_theta

def get_actions(node):
    adjacent_nodes = []
    for degrees in [-90, -60, -30, 0, 30, 60, 90]:
        adj_node = move_robot(node, degrees)
        x, y = adj_node[0], adj_node[1]
        if 0 < x < map_width and 0 < y < map_height:
            if map_data[y][x] != 1:
                adjacent_nodes.append(adj_node)
    return adjacent_nodes

def visualize_path():
    start_x, start_y = convert_coords(start_coords[0], start_coords[1])
    end_x, end_y = convert_coords(end_coords[0], end_coords[1])
    node_count = 0
    global map_image

    for visited_node in combined_visited_list:
        node_count += 1

        xn, yn = convert_coords(visited_node[0], visited_node[1])
        map_image[yn, xn] = (150, 150, 150)
        angle = visited_node[2]
        arrow_x = int(visited_node[0] + step_length * math.cos(np.pi + math.radians(angle)))
        arrow_y = int(visited_node[1] + step_length * math.sin(np.pi + math.radians(angle)))
        arrow_x, arrow_y = convert_coords(arrow_x, arrow_y)
        map_image = cv2.arrowedLine(map_image, (arrow_x, arrow_y), (xn, yn), (0, 200, 0), 1, tipLength=0.1)

        cv2.circle(map_image, (end_x, end_y), 6, (50, 50, 255), -1)
        cv2.circle(map_image, (start_x, start_y), 6, (50, 255, 50), -1)

        if node_count % 20 == 0:
            cv2.imshow("Map", map_image)
            cv2.waitKey(1)

    for node, angle in zip(robot_path, robot_angles):
        xn, yn = convert_coords(node[0], node[1])
        map_image[yn, xn] = (150, 150, 150)
        arrow_x = int(node[0] + step_length * math.cos(np.pi + math.radians(angle)))
        arrow_y = int(node[1] + step_length * math.sin(np.pi + math.radians(angle)))
        arrow_x, arrow_y = convert_coords(arrow_x, arrow_y)
        map_image = cv2.arrowedLine(map_image, (arrow_x, arrow_y), (xn, yn), (255, 0, 0), 2, tipLength=0.1)

        xn, yn = convert_coords(node[0], node[1])
        cv2.circle(map_image, (xn, yn), 1, (0, 255, 0), -1)
        cv2.imshow("Map", map_image)
        cv2.waitKey(1)
        time.sleep(0.05)

    cv2.imshow("Map", map_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def euclidean_dist(node1, node2):
   d = math.sqrt((node2[0] - node1[0]) ** 2 + (node2[1] - node1[1]) ** 2)
   return round(d, 1)

def find_path():
   global goal_node, goal_reached
   infinite = float('inf')

   while not forward_queue.empty():
       present_cost, node = forward_queue.get()
       node_coords = node[0]
       node_orientation = node[1]
       current_edge_cost = forward_edge_cost.get(node_coords, infinite)

       for node_v in backward_visited_list:
           nx, ny, nt = node_v
           if (nx, ny) == node_coords:
               if abs(nt - node[1]) % 180 == 0:
                   goal_node = node
                   print("Goal Reached!")
                   goal_reached = True
                   break

       if goal_reached:
           break

       adjacent_nodes = get_actions(node)

       for adj_node in adjacent_nodes:
           adj_node_coords = adj_node[:2]
           added_edge_cost = euclidean_dist(node_coords, adj_node_coords)
           updated_edge_cost = forward_edge_cost[node_coords] + added_edge_cost
           heuristic_cost = euclidean_dist(adj_node_coords, end_coords) * weight_factor
           total_cost = heuristic_cost + updated_edge_cost

           if not check_visited(adj_node, 'forward') or total_cost < forward_total_cost.get(adj_node_coords,
                                                                                             float('inf')):
               forward_edge_cost[adj_node_coords] = updated_edge_cost
               forward_total_cost[adj_node_coords] = total_cost
               lowest_edge_cost = total_cost
               orientation = ((360 + adj_node[2]) % 360) / 30

               for x in range(int(node_coords[0]), int(adj_node[0])):
                   for y in range(int(node_coords[1]), int(adj_node[1])):
                       forward_visited[y][x][int(orientation)] = 1

               forward_queue.put((lowest_edge_cost, (adj_node_coords, adj_node[2])))
               forward_parent[adj_node_coords] = node_coords
               forward_visited_list.append(adj_node)
               mark_visited(adj_node, 'forward')

       present_cost, node = backward_queue.get()
       node_coords = node[0]
       node_orientation = node[1]
       current_edge_cost = backward_edge_cost.get(node_coords, infinite)

       for node_v in forward_visited_list:
           nx, ny, nt = node_v
           if (nx, ny) == node_coords:
               if abs(nt - node[1]) % 180 == 0:
                   goal_node = node
                   print("Goal Reached!")
                   goal_reached = True
                   break

       if goal_reached:
           break

       adjacent_nodes = get_actions(node)

       for adj_node in adjacent_nodes:
           adj_node_coords = adj_node[:2]
           added_edge_cost = euclidean_dist(node_coords, adj_node_coords)
           updated_edge_cost = backward_edge_cost[node_coords] + added_edge_cost
           heuristic_cost = euclidean_dist(adj_node_coords, start_coords) * weight_factor
           total_cost = heuristic_cost + updated_edge_cost

           if not check_visited(adj_node, 'backward') or total_cost < backward_total_cost.get(adj_node_coords,
                                                                                               float('inf')):
               backward_edge_cost[adj_node_coords] = updated_edge_cost
               backward_total_cost[adj_node_coords] = total_cost
               lowest_edge_cost = total_cost
               orientation = ((360 + adj_node[2]) % 360) / 30

               for x in range(int(node_coords[0]), int(adj_node[0])):
                   for y in range(int(node_coords[1]), int(adj_node[1])):
                       backward_visited[y][x][int(orientation)] = 1

               backward_queue.put((lowest_edge_cost, (adj_node_coords, adj_node[2])))
               backward_parent[adj_node_coords] = node_coords
               backward_visited_list.append(adj_node)
               mark_visited(adj_node, 'backward')

   for node in forward_visited_list:
       x, y, angle = node
       forward_angle_dict[(x, y)] = angle

   forward_angle_dict[(start_coords[0], start_coords[1])] = start_orientation

   for node in backward_visited_list:
       x, y, angle = node
       backward_angle_dict[(x, y)] = angle

   backward_angle_dict[(end_coords[0], end_coords[1])] = end_orientation

   robot_path = []
   robot_angles = []
   node = goal_node
   node_coords = node[0]

   while node_coords != start_coords:
       robot_path.append(node_coords)
       node_coords = forward_parent[node_coords]

   robot_path.append(start_coords)
   robot_path.reverse()

   for node in robot_path:
       robot_angles.append(forward_angle_dict[node])

   reverse_robot_path = []
   reverse_robot_angles = []
   node = goal_node
   node_coords = node[0]

   while node_coords != end_coords:
       reverse_robot_path.append(node_coords)
       node_coords = backward_parent[node_coords]

   reverse_robot_path.append(end_coords)
   reverse_robot_path.reverse()

   for node in reverse_robot_path:
    reverse_robot_angles.append(backward_angle_dict[node])

   ang_new = []
   for a in reverse_robot_angles:
       new_a = ((a + 180) % 360)
       ang_new.append(new_a)

   reverse_robot_path.reverse()
   reverse_robot_path.pop(0)
   reverse_robot_angles.reverse()
#    try:
   reverse_robot_angles.pop()
#    except:
#        print("issue with reverse_robot_angles")

   robot_path.extend(reverse_robot_path)
   robot_angles.extend(ang_new)

   combined_visited_list = []

   forward_length = len(forward_visited_list)
   backward_length = len(backward_visited_list)

   if forward_length >= backward_length:
       greater_length = forward_length
   else:
       greater_length = backward_length

   for iteration in range(greater_length):
       if iteration < forward_length:
           combined_visited_list.append(forward_visited_list[iteration])

       if iteration < backward_length:
           combined_visited_list.append(backward_visited_list[iteration])

   return robot_path, robot_angles, combined_visited_list

# Call the functions to find the path and visualize
robot_path, robot_angles, combined_visited_list = find_path()
visualize_path()