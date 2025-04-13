import numpy as np
import matplotlib.pyplot as plt

# Define points for the first three letters of "SAI" - same as q2_kmeans.py
# Letter S (centered at x=5, with width 10 and height 15)
S_points = np.array([
    [10, 15],    # S1
    [5, 15],     # S2
    [0, 12],     # S3
    [5, 7.5],    # S4
    [10, 3],     # S5
    [5, 0],      # S6
    [0, 0],      # S7
])

# Letter A (centered at x=25, with width 10 and height 15)
A_points = np.array([
    [20, 0],     # A1
    [22, 8],     # A2
    [24, 15],    # A3
    [26, 15],    # A4
    [28, 8],     # A5
    [30, 0],     # A6
    [25, 8]      # A7
])

# Letter I (centered at x=45, with width 10 and height 15)
I_points = np.array([
    [40, 0],     # I1 - bottom left
    [50, 0],     # I2 - bottom right
    [45, 0],     # I3 - bottom center
    [45, 7.5],   # I4 - middle
    [45, 15],    # I5 - top center
    [40, 15],    # I6 - top left
    [50, 15]     # I7 - top right
])

# Combine all points
all_points = np.vstack([S_points, A_points, I_points])

# Assign labels to each point
point_labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
               'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
               'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def dbscan(points, epsilon, min_samples):
    """
    DBSCAN clustering algorithm
    
    Parameters:
    - points: numpy array of data points
    - epsilon: maximum distance between two points for neighborhood
    - min_samples: minimum number of points required to form a core point
    
    Returns:
    - labels: cluster labels for each point (-1 for noise)
    - core_points: boolean array indicating which points are core points
    - border_points: boolean array indicating which points are border points
    - neighborhoods: list of neighborhoods for each point
    """
    n_points = len(points)
    labels = np.full(n_points, -1)  # Initialize all points as noise
    cluster_id = 0
    
    # Step 1: Compute the neighborhood of each point
    neighborhoods = []
    core_points = np.zeros(n_points, dtype=bool)
    border_points = np.zeros(n_points, dtype=bool)
    
    print(f"\nStep 1: Computing neighborhoods with epsilon={epsilon}")
    for i in range(n_points):
        neighborhood = []
        for j in range(n_points):
            distance = euclidean_distance(points[i], points[j])
            if distance <= epsilon:
                neighborhood.append(j)
                
        neighborhoods.append(neighborhood)
        print(f"Point {point_labels[i]}: Neighborhood size = {len(neighborhood)}, " +
              f"Points = {[point_labels[j] for j in neighborhood]}")
        
        # Step 2: Identify core points and border points
        if len(neighborhood) >= min_samples:
            core_points[i] = True
    
    print(f"\nStep 2: Identifying core points (min_samples={min_samples})")
    for i in range(n_points):
        if core_points[i]:
            print(f"Point {point_labels[i]} is a CORE point with {len(neighborhoods[i])} neighbors")
        else:
            # Check if the point is a border point (connected to a core point)
            is_border = False
            for j in neighborhoods[i]:
                if core_points[j] and j != i:
                    border_points[i] = True
                    is_border = True
                    break
            
            if is_border:
                print(f"Point {point_labels[i]} is a BORDER point")
            else:
                print(f"Point {point_labels[i]} is a NOISE point")
    
    # Step 3: Expand clusters from core points
    print("\nStep 3: Expanding clusters from core points")
    for i in range(n_points):
        if not core_points[i] or labels[i] != -1:
            continue
        
        # Start a new cluster
        cluster_id += 1
        labels[i] = cluster_id
        print(f"Starting new cluster {cluster_id} from point {point_labels[i]}")
        
        # Process queue for connected core points
        queue = neighborhoods[i].copy()
        processed = set([i])
        
        while queue:
            j = queue.pop(0)
            if j in processed:
                continue
            
            processed.add(j)
            
            # Mark this point with current cluster
            if labels[j] == -1:  # Only if it's not already assigned to a cluster
                labels[j] = cluster_id
                print(f"  - Adding point {point_labels[j]} to cluster {cluster_id}")
                
                # If it's also a core point, add its neighbors to the queue
                if core_points[j]:
                    for neighbor in neighborhoods[j]:
                        if neighbor not in processed:
                            queue.append(neighbor)
    
    return labels, core_points, border_points, neighborhoods

def visualize_dbscan(points, labels, core_points, border_points, epsilon, min_samples, iteration=None):
    plt.figure(figsize=(15, 10))
    
    # Assign colors based on clusters
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot points
    for i, label in enumerate(unique_labels):
        if label == -1:
            # Noise points in black
            cluster_points = points[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='black', marker='x', s=100, label='Noise')
        else:
            # Cluster points in colors
            cluster_points = points[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], s=100, label=f'Cluster {label}')
    
    # Highlight core points with a larger circle
    core_indices = np.where(core_points)[0]
    plt.scatter(points[core_indices, 0], points[core_indices, 1], s=200, edgecolors='black', facecolors='none', linewidths=2, label='Core Point')
    
    # Highlight border points with a different symbol
    border_indices = np.where(border_points)[0]
    plt.scatter(points[border_indices, 0], points[border_indices, 1], s=200, edgecolors='black', facecolors='none', marker='s', linewidths=2, label='Border Point')
    
    # Label each point
    for i, (x, y) in enumerate(points):
        plt.annotate(f"{point_labels[i]}", (x, y), fontsize=10, xytext=(5, 5), textcoords='offset points')
    
    # Draw epsilon circles around ALL core points
    if len(core_indices) > 0:
        for idx in core_indices:
            circle = plt.Circle((points[idx][0], points[idx][1]), epsilon, fill=False, linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)
            # For clarity, only add the epsilon label to a few points to avoid overcrowding
            if idx in core_indices[:min(3, len(core_indices))]:
                plt.annotate(f"ε={epsilon}", (points[idx][0], points[idx][1]-1), fontsize=8, ha='center')
    
    title = f'DBSCAN Clustering (ε={epsilon}, min_samples={min_samples})'
    if iteration is not None:
        title += f' - Parameter Set {iteration}'
    plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Original plot to visualize the raw data
plt.figure(figsize=(15, 10))
plt.scatter(all_points[:, 0], all_points[:, 1], c='blue', s=100)

# Label each point
for i, (x, y) in enumerate(all_points):
    plt.annotate(point_labels[i], (x, y), fontsize=12)

# Connect the points to form the letter shapes
# Letter S
plt.plot([S_points[0][0], S_points[1][0], S_points[2][0], S_points[3][0], 
          S_points[4][0], S_points[5][0], S_points[6][0]], 
         [S_points[0][1], S_points[1][1], S_points[2][1], S_points[3][1], 
          S_points[4][1], S_points[5][1], S_points[6][1]], 'r-')

# Letter A
plt.plot([A_points[0][0], A_points[1][0], A_points[2][0], A_points[3][0], 
          A_points[4][0], A_points[5][0]], 
         [A_points[0][1], A_points[1][1], A_points[2][1], A_points[3][1], 
          A_points[4][1], A_points[5][1]], 'r-')
plt.plot([A_points[1][0], A_points[6][0], A_points[4][0]], 
         [A_points[1][1], A_points[6][1], A_points[4][1]], 'r-')  # Connect A2->A7->A5

# Letter I
plt.plot([I_points[5][0], I_points[6][0]], [I_points[5][1], I_points[6][1]], 'r-')  # top horizontal
plt.plot([I_points[0][0], I_points[1][0]], [I_points[0][1], I_points[1][1]], 'r-')  # bottom horizontal
plt.plot([I_points[2][0], I_points[4][0]], [I_points[2][1], I_points[4][1]], 'r-')  # vertical line

plt.title('Letter Representation in Cartesian Plane')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')
plt.show()

# Try different parameter combinations
print("=== DBSCAN Clustering Analysis ===")
print("==================================")

# Parameter sets to try (epsilon, min_samples)
parameter_sets = [
    (8,6),
    (9,6),
    # (9.1,6),
    (9,2),
    (9.1,5),
    (10,5),
    (10,6),
    (10,7),
    (10,8)
]

# Run DBSCAN with different parameter combinations
for i, (epsilon, min_samples) in enumerate(parameter_sets, 1):
    print(f"\n\nParameter Set {i}: epsilon={epsilon}, min_samples={min_samples}")
    print("="*50)
    
    labels, core_points, border_points, neighborhoods = dbscan(all_points, epsilon, min_samples)
    
    # Count the number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print("\nDBSCAN Summary:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Print cluster assignments
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            cluster_label = "Noise"
        else:
            cluster_label = f"Cluster {cluster_id}"
        
        cluster_points_indices = np.where(labels == cluster_id)[0]
        print(f"{cluster_label}: {[point_labels[i] for i in cluster_points_indices]}")
    
    # Visualize the results
    visualize_dbscan(all_points, labels, core_points, border_points, epsilon, min_samples, i)

