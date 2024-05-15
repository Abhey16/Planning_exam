import matplotlib.pyplot as plt
import numpy as np

# Define functions
def calculate_attractive_force(agent_pos, target_pos, target_charge, agent_charge, target_threshold):
    direction = target_pos - agent_pos
    distance = np.linalg.norm(direction)
    if distance <= target_threshold:
        return np.zeros(2)
    force = target_charge * agent_charge / (distance ** 2)
    return force * direction / distance

def calculate_repulsive_force(agent_pos, obstacle_pos, obstacle_radius, obstacle_charge, agent_charge, obstacle_threshold):
    direction = agent_pos - obstacle_pos
    distance = np.linalg.norm(direction)
    if distance <= obstacle_radius + obstacle_threshold:
        if distance <= obstacle_radius:
            distance = obstacle_radius
        force = obstacle_charge * agent_charge / (distance ** 2)
        return force * direction / distance
    else:
        return np.zeros(2)

def potential_field_path_planning(start_point, end_point, shapes, canvas_size, agent_charge, target_charge, obstacle_charge, step_length, target_threshold, obstacle_threshold):
    canvas_width, canvas_height = canvas_size
    agent_pos = np.array([start_point['x'], start_point['y']], dtype=float)
    path = [agent_pos]
    exploration_nodes = []

    max_iterations = 1000
    iteration = 0

    while np.linalg.norm(agent_pos - np.array([end_point['x'], end_point['y']])) > target_threshold and iteration < max_iterations:
        # Calculate attractive force 
        attractive_force_vec = calculate_attractive_force(agent_pos, np.array([end_point['x'], end_point['y']]), target_charge, agent_charge, target_threshold)
        
        # Calculate repulsive forces
        repulsive_force_vec = np.zeros(2)
        for shape in shapes:
            repulsive_force_vec += calculate_repulsive_force(agent_pos, np.array([shape['x'], shape['y']]), shape['size'] / 2, obstacle_charge, agent_charge, obstacle_threshold)
        
        # Calculate the resultant force
        resultant_force = attractive_force_vec + repulsive_force_vec
        
        if np.linalg.norm(resultant_force) > 0:
            resultant_force /= np.linalg.norm(resultant_force)
        
        new_agent_pos = agent_pos + step_length * resultant_force
        path.append(new_agent_pos)
        exploration_nodes.append(new_agent_pos)

        agent_pos = new_agent_pos
        
        iteration += 1

    return path, exploration_nodes

def visualize_path(path, start_point, end_point, shapes, canvas_size):
    canvas_width, canvas_height = canvas_size
    scale_factor = 10
    scaled_width = canvas_width // scale_factor
    scaled_height = canvas_height // scale_factor

    fig = plt.figure(figsize=(scaled_width/100, scaled_height/100), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_xticks(range(0, canvas_width+1, 1000))
    ax.set_yticks(range(0, canvas_height+1, 500))

    fig.suptitle(f'potential force-field implementation')

    for shape in shapes:
        shape_radius = shape['size'] / 2
        shape_plot = plt.Circle((shape['x'], shape['y']), shape_radius, color='red')
        ax.add_patch(shape_plot)

    plt.scatter(start_point['x'], start_point['y'], color='g', marker='o', s=100, label='Start Point')
    plt.scatter(end_point['x'], end_point['y'], color='r', marker='o', s=100, label='End Point')

    path_x, path_y = zip(*path)
    plt.plot(path_x, path_y, 'b-', linewidth=2)

    plt.show()

if __name__ == '__main__':

    # Set parameters
    canvas_size = (6000, 3000)
    start_point = {'x': 0, 'y': 1500}
    end_point = {'x': 6000, 'y': 1500}
    shapes = [
        {'x': 1120, 'y': 2425, 'size': 800},
        {'x': 2630, 'y': 900, 'size': 1400},
        {'x': 4450, 'y': 2200, 'size': 750}
    ]
    agent_charge = 1.0
    target_charge = 5.0
    obstacle_charge = 100.0
    step_length = 100
    target_threshold = 100
    obstacle_threshold = 200

    # Run path planning
    path, exploration_nodes = potential_field_path_planning(start_point, end_point, shapes, canvas_size, agent_charge, target_charge, obstacle_charge, step_length, target_threshold, obstacle_threshold)

    # Visualize the path
    visualize_path(path, start_point, end_point, shapes, canvas_size)