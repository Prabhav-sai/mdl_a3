import numpy as np
import matplotlib.pyplot as plt

# Define points for the first three letters of "SAI"
# Each letter is represented by 7 points

# Letter S (centered at x=5, with width 10 and height 15)
S_points = np.array([
    [10, 15],    # S1
    [5, 15],     # S2
    [0, 12],     # S3
    [5, 7.5],    # S4 - fixed missing comma
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
    [45, 7.5],   # middle
    [45, 15],    # top center
    [40, 15],    # top left
    [50, 15]     # top right
])

# Combine all points
all_points = np.vstack([S_points, A_points, I_points])

# Assign labels to each point
point_labels = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7',
               'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
               'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7']

# Plot the points
plt.figure(figsize=(15, 10))
plt.scatter(all_points[:, 0], all_points[:, 1], c='blue', s=100)

# Label each point
for i, (x, y) in enumerate(all_points):
    plt.annotate(point_labels[i], (x, y), fontsize=12)

# Connect the points to form the letter shapes
# Letter S - connect points consecutively S1->S2->S3->...->S7
plt.plot([S_points[0][0], S_points[1][0], S_points[2][0], S_points[3][0], 
          S_points[4][0], S_points[5][0], S_points[6][0]], 
         [S_points[0][1], S_points[1][1], S_points[2][1], S_points[3][1], 
          S_points[4][1], S_points[5][1], S_points[6][1]], 'r-')

# Letter A - connect A1->A2->A3->A4->A5->A6, then A2->A7->A5
plt.plot([A_points[0][0], A_points[1][0], A_points[2][0], A_points[3][0], 
          A_points[4][0], A_points[5][0]], 
         [A_points[0][1], A_points[1][1], A_points[2][1], A_points[3][1], 
          A_points[4][1], A_points[5][1]], 'r-')
plt.plot([A_points[1][0], A_points[6][0], A_points[4][0]], 
         [A_points[1][1], A_points[6][1], A_points[4][1]], 'r-')  # Connect A2->A7->A5

# Letter I - draw horizontal top, bottom and vertical line
plt.plot([I_points[5][0], I_points[6][0]], [I_points[5][1], I_points[6][1]], 'r-')  # top horizontal
plt.plot([I_points[0][0], I_points[1][0]], [I_points[0][1], I_points[1][1]], 'r-')  # bottom horizontal
plt.plot([I_points[2][0], I_points[4][0]], [I_points[2][1], I_points[4][1]], 'r-')  # vertical line

plt.title('Letter Representation in Cartesian Plane')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.axis('equal')
plt.show()

# Manual K-Means Clustering
print("Manual K-Means Clustering Analysis")
print("=================================")

# Let's assume k=3 (one cluster for each letter)
k = 3

# Step 1: Choose initial centroids (one from each letter)
centroids = np.array([S_points[1], A_points[3], I_points[6]])


print("Initial Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Perform K-Means clustering manually
max_iterations = 10
current_centroids = centroids.copy()

for iteration in range(1, max_iterations + 1):
    print(f"\nIteration {iteration}:")
    
    # Step 2: Assign each point to the nearest centroid
    clusters = [[] for _ in range(k)]
    cluster_indices = []
    point_distances = []  # Store distances for visualization
    
    print("Assigning points to clusters...")
    for i, point in enumerate(all_points):
        distances = [euclidean_distance(point, centroid) for centroid in current_centroids]
        closest_centroid = np.argmin(distances)
        cluster_indices.append(closest_centroid)
        clusters[closest_centroid].append(i)
        point_distances.append(distances[closest_centroid])  # Store the distance to assigned centroid
        
        print(f"Point {point_labels[i]} {point}: Distances = {distances}, Assigned to cluster {closest_centroid+1}")
    
    # Plot the current iteration's clustering BEFORE updating centroids
    plt.figure(figsize=(15, 10))
    
    colors = ['red', 'green', 'blue']
    for i in range(k):
        cluster_points = all_points[[idx for idx, cluster_idx in enumerate(cluster_indices) if cluster_idx == i]]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}', s=100)
    
    # Draw current centroids (before update)
    plt.scatter(current_centroids[:, 0], current_centroids[:, 1], c='black', marker='X', s=200, label='Current Centroids')
    
    # Label the centroids
    for i, (x, y) in enumerate(current_centroids):
        plt.annotate(f"Centroid {i+1}\n({x:.2f}, {y:.2f})", 
                    (x, y), 
                    fontsize=8,  # Smaller font size
                    color='black',
                    weight='bold',
                    xytext=(0, -30),  # Position further below the centroid
                    textcoords='offset points',
                    ha='center')  # Horizontally centered
    
    # Label each point with point name and distance to assigned centroid
    for i, (x, y) in enumerate(all_points):
        centroid_idx = cluster_indices[i]
        plt.annotate(f"{point_labels[i]}\nd={point_distances[i]:.2f}", 
                    (x, y), 
                    fontsize=8,  # Smaller font size
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.title(f'K-Means Clustering - Iteration {iteration} (Before Centroid Update)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # Step 3: Recalculate centroids
    old_centroids = current_centroids.copy()
    
    print("\nUpdating centroids...")
    for i, cluster in enumerate(clusters):
        if cluster:  # If cluster is not empty
            cluster_points = all_points[cluster]
            new_centroid = np.mean(cluster_points, axis=0)
            current_centroids[i] = new_centroid
            print(f"Centroid {i+1} updated to {new_centroid} (mean of points {[point_labels[j] for j in cluster]})")
    
    # Optional: Plot again after updating centroids to see the difference
    plt.figure(figsize=(15, 10))
    
    for i in range(k):
        cluster_points = all_points[[idx for idx, cluster_idx in enumerate(cluster_indices) if cluster_idx == i]]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}', s=100)
    
    # Draw updated centroids
    plt.scatter(current_centroids[:, 0], current_centroids[:, 1], c='black', marker='X', s=200, label='Updated Centroids')
    
    # Label the updated centroids
    for i, (x, y) in enumerate(current_centroids):
        plt.annotate(f"Centroid {i+1}\n({x:.2f}, {y:.2f})", 
                    (x, y), 
                    fontsize=8,  # Smaller font size
                    color='black',
                    weight='bold',
                    xytext=(0, -30),  # Position further below the centroid
                    textcoords='offset points',
                    ha='center')  # Horizontally centered
    
    # Label points same as before
    for i, (x, y) in enumerate(all_points):
        centroid_idx = cluster_indices[i]
        plt.annotate(f"{point_labels[i]}\nd={point_distances[i]:.2f}", 
                    (x, y), 
                    fontsize=8,  # Smaller font size
                    xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.title(f'K-Means Clustering - Iteration {iteration} (After Centroid Update)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
    
    # Step 4: Check for convergence
    converged = True
    for i in range(k):
        if euclidean_distance(old_centroids[i], current_centroids[i]) > 1e-4:
            converged = False
            break
            
    if converged:
        print("\nK-Means has converged!")
        break

# Print final cluster assignments
print("\nFinal Cluster Assignments:")
for i in range(k):
    cluster_points_indices = [idx for idx, cluster_idx in enumerate(cluster_indices) if cluster_idx == i]
    print(f"Cluster {i+1}: {[point_labels[idx] for idx in cluster_points_indices]}")
