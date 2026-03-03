import h5py
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Path to the saved HDF5 file
output_hdf5_file = '/cs/projects/MWzeronoise/Analysis/4Shivangi/Results/RF_VR_mapping/RFarea_stim/20230214/RF_stim_area.h5'
save_path = '/cs/projects/MWzeronoise/Analysis/4Shivangi/plots/RF VR mapping/RFarea_VR_plot.pdf'

def process_file(file_path, max_trials=None, max_time_points=None):
    """
    Processes the HDF5 file to calculate total and inside counts for test points,
    with optional limits on the number of trials and time points processed.
    
    Args:
        file_path (str): Path to the HDF5 file.
        max_trials (int): Maximum number of trials to process. None for all trials.
        max_time_points (int): Maximum number of time points per trial. None for all time points.

    Returns:
        total_counts (dict): Total counts of test points across time points.
        inside_a_counts (dict): Counts of test points inside StimA outline.
        inside_b_counts (dict): Counts of test points inside StimB outline.
    """
    total_counts = defaultdict(int)
    inside_a_counts = defaultdict(int)
    inside_b_counts = defaultdict(int)

    with h5py.File(file_path, 'r') as file:

        # Loop through trials, limit to max_trials if specified
        for trial_index, trial_name in enumerate(file.keys()):
            if max_trials is not None and trial_index >= max_trials:
                break  # Stop if trial limit is reached
            
            trial_group = file[trial_name]

            # Loop through time points, limit to max_time_points if specified
            for time_point_index, time_point_name in enumerate(trial_group.keys()):
                if max_time_points is not None and time_point_index >= max_time_points:
                    break  # Stop if time point limit is reached
                
                point_index = 0  # Initialize a counter for point numbering
                time_point_group = trial_group[time_point_name]

                # Loop through all test points in the time point
                for point_name in time_point_group.keys():
                    point_group = time_point_group[point_name]

                    # Extract data
                    #test_point = tuple(point_group['test_point'][()])  # Convert to tuple for dictionary key
                    inside_a = point_group['inside_transformed_outline_A'][()]
                    inside_b = point_group['inside_transformed_outline_B'][()]

                    # Assign a unique string identifier for each test point
                    point_id = f'Point_{point_index}'
                    point_index += 1

                    # Update total counts and inside counts for point_id
                    total_counts[point_id] += 1
                    inside_a_counts[point_id] += int(inside_a)
                    inside_b_counts[point_id] += int(inside_b)
                        
    return total_counts, inside_a_counts, inside_b_counts


def plot_percent_inside(file_path, save_path, max_trials=None, max_time_points=None):
    """
    Calculates percentages and generates a bar plot for test points, with subplots of 32 points each.
    Saves the plot as a PDF.
    
    Args:
        file_path (str): Path to the HDF5 file.
        save_path (str): Path where the plot will be saved.
        max_trials (int): Maximum number of trials to process. None for all trials.
        max_time_points (int): Maximum number of time points per trial. None for all time points.
    """
    try:
        # Process the HDF5 file
        total_counts, inside_a_counts, inside_b_counts = process_file(
            file_path, max_trials=max_trials, max_time_points=max_time_points
        )

        # Compute percentages for all test points
        percent_inside_a = {point: (inside_a_counts[point] / total_counts[point]) * 100 for point in total_counts}
        percent_inside_b = {point: (inside_b_counts[point] / total_counts[point]) * 100 for point in total_counts}

        # Prepare data for plotting
        points = list(total_counts.keys())
        percentages_a = [percent_inside_a[point] for point in points]
        percentages_b = [percent_inside_b[point] for point in points]

        # Group the data into chunks of 32 points each
        chunk_size = 32
        num_chunks = (len(points) + chunk_size - 1) // chunk_size  # Calculate number of subplots

        fig, axes = plt.subplots(num_chunks, 1, figsize=(12, 6 * num_chunks), squeeze=False)
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(points))
            chunk_points = points[start_idx:end_idx]
            chunk_percentages_a = percentages_a[start_idx:end_idx]
            chunk_percentages_b = percentages_b[start_idx:end_idx]

            # Create bar plot for this chunk
            x = np.arange(len(chunk_points))
            width = 0.35
            ax.bar(x - width / 2, chunk_percentages_a, width, label='InsideOutlineA (%)')
            ax.bar(x + width / 2, chunk_percentages_b, width, label='InsideOutlineB (%)')

            # Formatting the subplot
            ax.set_xlabel('Test Points')
            ax.set_ylabel('Percent of Time')
            ax.set_title(f'Percentage of Time Test Points Are Inside Outlines (Points {start_idx + 1} to {end_idx})')
            ax.set_xticks(x)
            ax.set_xticklabels(chunk_points, rotation=45, ha='right')
            ax.legend()

        # Adjust layout and save the plot
        plt.tight_layout()
        #plt.show()
        plt.savefig(save_path, format='pdf')

        print(f"Plot saved successfully to: {save_path}")
        
    except Exception as e:
        print(f"Error processing the file: {e}")

# Example Usage: Test on limited trials and time points
plot_percent_inside(output_hdf5_file, save_path, max_trials=None, max_time_points=None)
