import csv

import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

# ===== Analysis Settings =====
# Graph drawing settings
AXIS_LABEL_FONTSIZE = 28  # Axis label font size (12-14pt recommended)
TICK_LABEL_FONTSIZE = 24  # Tick label font size (10-12pt recommended)
LEGEND_FONTSIZE = 20  # Legend font size
# ===================

# CSV file names
csv_optimal_bandwidth = "./simulation_result/log_optimal_bandwidth.csv"
csv_aco_avg_bandwidth = "./simulation_result/log_aco_avg_bandwidth.csv"
export_image_svg = "./simulation_result/result_bandwidth_comparison.svg"
export_image_pdf = "./simulation_result/result_bandwidth_comparison.pdf"


def read_csv_data(file_path):
    """
    Read CSV data and return each simulation's data as a list.
    Each row is one simulation, each column is one generation's bandwidth value.
    """
    data = []
    try:
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Skip empty rows
                    data.append([float(val) for val in row])
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []

    if not data:
        print(f"Warning: CSV file '{file_path}' is empty.")
        return []

    return data


def calculate_average_bandwidth(data):
    """
    Calculate average bandwidth for each generation.
    data: List of lists (each row is a simulation, each column is a generation)
    """
    if not data:
        return []

    num_generations = len(data[0])

    average_bandwidth = []
    for gen_idx in range(num_generations):
        # Aggregate bandwidth values for that generation from all simulations
        generation_bandwidths = [row[gen_idx] for row in data if len(row) > gen_idx]
        if generation_bandwidths:
            avg = sum(generation_bandwidths) / len(generation_bandwidths)
            average_bandwidth.append(avg)
        else:
            average_bandwidth.append(0)

    return average_bandwidth


def calculate_statistics(data):
    """
    Calculate statistics (mean, standard deviation) for each generation.
    """
    if not data:
        return [], []

    num_generations = len(data[0])
    averages = []
    std_devs = []

    for gen_idx in range(num_generations):
        generation_bandwidths = [row[gen_idx] for row in data if len(row) > gen_idx]
        if generation_bandwidths:
            avg = np.mean(generation_bandwidths)
            std = np.std(generation_bandwidths)
            averages.append(avg)
            std_devs.append(std)
        else:
            averages.append(0)
            std_devs.append(0)

    return averages, std_devs


# Load data
optimal_data = read_csv_data(csv_optimal_bandwidth)
aco_data = read_csv_data(csv_aco_avg_bandwidth)

if optimal_data and aco_data:
    # Calculate average bandwidth for each generation
    optimal_avg, optimal_std = calculate_statistics(optimal_data)
    aco_avg, aco_std = calculate_statistics(aco_data)

    x_values = list(range(len(optimal_avg)))

    # Draw graph (standard paper format: box type)
    plt.figure(figsize=(10, 7))  # Aspect ratio close to silver ratio

    # Optimal solution plot (black solid line)
    plt.plot(
        x_values,
        optimal_avg,
        label="Optimal Solution (Modified Dijkstra)",
        color="black",
        linewidth=2.5,
        linestyle="-",
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
    )

    # Optimal solution standard deviation shading
    optimal_upper = [avg + std for avg, std in zip(optimal_avg, optimal_std)]
    optimal_lower = [avg - std for avg, std in zip(optimal_avg, optimal_std)]
    plt.fill_between(
        x_values,
        optimal_lower,
        optimal_upper,
        alpha=0.15,
        color="gray",
        edgecolor="none",
    )

    # ACO average bandwidth plot (dark gray dashed line)
    plt.plot(
        x_values,
        aco_avg,
        label="ACO Average Bandwidth",
        color="dimgray",
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=4,
        markerfacecolor="dimgray",
        markeredgecolor="dimgray",
        markeredgewidth=1.5,
    )

    # ACO standard deviation shading
    aco_upper = [avg + std for avg, std in zip(aco_avg, aco_std)]
    aco_lower = [avg - std for avg, std in zip(aco_avg, aco_std)]
    plt.fill_between(
        x_values,
        aco_lower,
        aco_upper,
        alpha=0.1,
        color="lightgray",
        edgecolor="none",
    )

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Generation", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Bottleneck Bandwidth [Mbps]", fontsize=AXIS_LABEL_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    # Standard paper format (box type: show all borders)
    ax = plt.gca()
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    # Set all borders to black with appropriate line width
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.5)

    # Tick settings
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONTSIZE,
        direction="out",
        length=6,
        width=1.5,
        color="black",
    )

    # Minor tick settings
    ax.tick_params(
        axis="both",
        which="minor",
        direction="out",
        length=3,
        width=1.0,
        color="black",
    )

    # Enable minor ticks
    ax.minorticks_on()

    plt.tight_layout()

    # Save as SVG
    plt.savefig(export_image_svg, format="svg")
    print(f"Graph (SVG) saved to {export_image_svg}")

    # Save as PDF
    plt.savefig(export_image_pdf, format="pdf")
    print(f"Graph (PDF) saved to {export_image_pdf}")

    # Display statistics
    print("\n=== Statistics ===")
    print(f"Number of simulations: {len(optimal_data)}")
    print(f"Number of generations: {len(optimal_avg)}")

    # Final generation statistics
    if optimal_avg and aco_avg:
        final_gen = len(optimal_avg) - 1
        print(f"\nFinal generation (generation {final_gen}):")
        print(f"  Optimal avg: {optimal_avg[-1]:.2f} Mbps (std: {optimal_std[-1]:.2f})")
        print(f"  ACO avg: {aco_avg[-1]:.2f} Mbps (std: {aco_std[-1]:.2f})")

        # Calculate how close ACO is to optimal
        if optimal_avg[-1] > 0:
            achievement_ratio = (aco_avg[-1] / optimal_avg[-1]) * 100
            print(f"  ACO achievement rate: {achievement_ratio:.2f}%")

    plt.show()
else:
    print("Error: Failed to load data.")
