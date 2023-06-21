import pso
import image_gen
import os
import time
import shutil
import numpy as np

n_pieces = 2  # 1 < n_pieces < 6

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative path
folder_path = os.path.join(script_dir, "0" + str(n_pieces) + "/")

image_path = folder_path + "image.png"

test_n = 1 # number of tests to generate
tries = 5 # number of times the algorithm will be run for each test
paper_width = 100
paper_height = 100
w = 0.7 # inertia weight
c1 = 1 # cognitive weight
c2 = 2 # social weight
desired_fitness=0.005 # value for fitness at which the solution is marked as found

for test_i in range(1,test_n+1):
    with open(folder_path + "batch_" + str(n_pieces) + "_results.csv", "a") as file:
        file.write(str(test_i) + "\n")
    if os.path.exists(folder_path + str(test_i)):
        shutil.rmtree(folder_path + str(test_i))
    os.makedirs(folder_path + str(test_i))
    image_folder = folder_path + str(test_i) + "/"
    image_gen.cut_image(image_path, image_folder, n_pieces)
    image_sizes = pso.get_image_dimensions_from_folder(image_folder)
    N = len(image_sizes) # number of images to be placed on the canvas
    dimensions = 3 * N # dimensions of the PSO
    bounds = (0, 100)  # bounds for initialisation of values
    paper_size = (paper_width, paper_height)
    best_positions = []
    best_fitnesses = []
    for try_i in range(1,tries+1):

        pso_object = pso.PSO(population_size=500, dimensions=dimensions, image_sizes=image_sizes, paper_size=paper_size, bounds=bounds, desired_fitness=desired_fitness, w=w, c1=c1, c2=c2)
        best_position = pso_object.run()

        with open(folder_path + "batch_" + str(n_pieces) + "_results.csv", "a") as file:
            if pso_object.iterations_without_improvement < 500:
                file.write(f"{try_i}, {pso_object.iterations}\n")
        best_positions.append(best_position.reshape(-1, 3))
        best_fitnesses.append(pso_object.gbest_fitness)

    with open(image_folder + "output.txt", "w") as file:
        positions = best_positions[np.argmin(best_fitnesses)]
        file.write(image_folder + "\n")
        for i, (x, y, scale) in enumerate(positions):
            file.write(f"Image {i}: x = {x}, y = {y}, scale = {scale}\n")


