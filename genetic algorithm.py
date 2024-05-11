import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_images(boothi_path, group_path):
    boothi = cv2.imread(boothi_path, cv2.IMREAD_GRAYSCALE)
    group = cv2.imread(group_path, cv2.IMREAD_GRAYSCALE)
    return boothi, group

def find_face(boothi, group, population_size=120, generations=150, mutation_rate=0.3):
    boothi_height, boothi_width = boothi.shape
    group_height, group_width = group.shape
    best_fitness = -1
    best_location = None
    fitnesses = []

    for _ in range(generations):
        # Generate random candidate locations
        candidates_x = np.random.randint(0, group_width - boothi_width + 1, size=population_size)
        candidates_y = np.random.randint(0, group_height - boothi_height + 1, size=population_size)

        # Evaluate fitness for each candidate
        fitness_values = []
        for x, y in zip(candidates_x, candidates_y):
            boothi_patch = group[y:y+boothi_height, x:x+boothi_width]
            if boothi_patch.shape != boothi.shape:
                fitness_values.append(-1)  # Invalid patch size
            else:
                corr = np.corrcoef(boothi.ravel(), boothi_patch.ravel())[0, 1]
                fitness_values.append(corr)
        
        # Store fitness statistics
        max_fitness = np.max(fitness_values)
        min_fitness = np.min(fitness_values)
        mean_fitness = np.mean(fitness_values)
        fitnesses.append((max_fitness, mean_fitness, min_fitness))

        # Find the best candidate
        max_fitness_index = np.argmax(fitness_values)
        if fitness_values[max_fitness_index] > best_fitness:
            best_fitness = fitness_values[max_fitness_index]
            best_location = (candidates_x[max_fitness_index], candidates_y[max_fitness_index])

        # Perform mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                candidates_x[i] = np.random.randint(max(0, candidates_x[i] - 10), min(group_width - boothi_width, candidates_x[i] + 10) + 1)
                candidates_y[i] = np.random.randint(max(0, candidates_y[i] - 10), min(group_height - boothi_height, candidates_y[i] + 10) + 1)

    return best_location, fitnesses

def draw_rectangle(image, location, boothi_shape):
    x, y = location
    boothi_width, boothi_height = boothi_shape
    return cv2.rectangle(image.copy(), (x, y), (x + boothi_width, y + boothi_height), (255, 255, 255), 2)

def plot_fitness_stats(fitnesses):
    max_fitnesses, mean_fitnesses, min_fitnesses = zip(*fitnesses)
    plt.plot(range(len(max_fitnesses)), max_fitnesses, label='Max Fitness')
    plt.plot(range(len(mean_fitnesses)), mean_fitnesses, label='Mean Fitness')
    plt.plot(range(len(min_fitnesses)), min_fitnesses, label='Min Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness vs Generation')
    plt.legend()
    plt.show()

def main(boothi_path, group_path):
    boothi, group = load_and_preprocess_images(boothi_path, group_path)
    best_location, fitnesses = find_face(boothi, group)
    max_fitness = max(fitness[0] for fitness in fitnesses)  # Get the maximum fitness value from all generations
    if best_location is not None:
        print(max_fitness)  # Print the maximum fitness value
        result_image = draw_rectangle(group, best_location, boothi.shape)
        plt.imshow(result_image, cmap='gray')
        plt.title(f"Best Fitness: {max_fitness}")
        plt.axis('off')
        plt.show()
        plot_fitness_stats(fitnesses)

        # Check if fitness threshold is reached
        if max_fitness >= 0.85:
            print("Face found with fitness >= 0.85")
            marked_image = draw_rectangle(group, best_location, boothi.shape)
            plt.imshow(marked_image, cmap='gray')
            plt.title("Face Found")
            plt.axis('off')
            plt.show()
    else:
        print("Face not found.")


if __name__ == "__main__":
    boothi_path = "boothiGray.jpg"
    group_path = "groupGray.jpg"
    main(boothi_path, group_path)
