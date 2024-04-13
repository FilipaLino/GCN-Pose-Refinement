import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot MPJPE values based on occlusion percentage or comparison.")
    parser.add_argument("--subject", type=str, required=True, help="Subject to analyze")
    parser.add_argument("--plot_type", type=str, required=True, choices=['occlusion', 'comparison', 'comparison-avg'], help="Type of plot to generate")
    return parser.parse_args()

# A function to compute the percentage of occluded keypoints for each pose
def occlusion_percentage(occluded_keypoints):
    num_keypoints = occluded_keypoints.shape[1]
    num_occluded = np.sum(occluded_keypoints == 0, axis=1)
    percentages = (num_occluded / num_keypoints) * 100
    return percentages

# Group poses by occlusion percentage and compute average MPJPE for each group
def average_mpjpe_by_occlusion(mpjpe_values, occlusion_percentages):
    unique_percentages = np.unique(occlusion_percentages)
    avg_mpjpe_by_occlusion = {}
    for percentage in unique_percentages:
        indices = np.where(occlusion_percentages == percentage)
        avg_mpjpe_by_occlusion[percentage] = np.mean(mpjpe_values[indices])
    return avg_mpjpe_by_occlusion

def main(args):
    # Specify the directory you want to list files from
    directory_path = f"../Synthetic_H3.6M/{args.subject}/Occlusions"
    # List all filenames in the directory
    actions = os.listdir(directory_path)

    # Load the .npz GCN file
    gcn = np.load(f'S{args.subject}_MPJPE/mpjpe_per_frame.npz', allow_pickle=True)

    # Load the .npz VideoPose3D file
    vp3d = np.load(f'../VideoPose3D/{args.subject}_MPJPE/mpjpe_per_frame.npz', allow_pickle=True)

    occl={}
    # Print all filenames
    for action in actions:
        aux=[]

        occlusions = np.load(os.path.join(directory_path, f"{action}/occluded_kpt.npz"), allow_pickle=True)
        for i in range(4):
            aux.append(occlusions[f'Cam_{i}'])
            '''
            for j in range(occlusions[f'Cam_{i}'].shape[0]):
               
                aux2=sum(occlusions[f'Cam_{i}'][j])
                if aux2<2:
                    print(action)
            
            '''
        occl[action] = np.concatenate(aux, axis=0)
    
    if args.plot_type == "occlusion":
        # Compute the occlusion percentages and average MPJPE values grouped by occlusion percentages for all actions
        results = {}
        # Aggregate MPJPE values and occlusion percentages across all actions
        all_occlusion_perc = []
        all_mpjpe_gcn = []
        all_mpjpe_vp3d = []

        for action in actions:
            occlusion_perc = occlusion_percentage(occl[action])
            avg_mpjpe_gcn = average_mpjpe_by_occlusion(gcn[action], occlusion_perc)
            avg_mpjpe_vp3d = average_mpjpe_by_occlusion(vp3d[action], occlusion_perc)
            results[action] = {'GCN': avg_mpjpe_gcn, 'VideoPose3D': avg_mpjpe_vp3d}

            all_occlusion_perc.extend(occlusion_perc)
            all_mpjpe_gcn.extend(gcn[action])
            all_mpjpe_vp3d.extend(vp3d[action])

        # Convert lists to numpy arrays for easier processing
        all_occlusion_perc = np.array(all_occlusion_perc)
        all_mpjpe_gcn = np.array(all_mpjpe_gcn)
        all_mpjpe_vp3d = np.array(all_mpjpe_vp3d)

        # Compute the global average MPJPE for each occlusion percentage
        global_avg_mpjpe_gcn = average_mpjpe_by_occlusion(all_mpjpe_gcn, all_occlusion_perc)
        global_avg_mpjpe_vp3d = average_mpjpe_by_occlusion(all_mpjpe_vp3d, all_occlusion_perc)

        # Plotting using a bar plot
        plt.figure(figsize=(12, 7))

        # Sorting the keys (occlusion percentages) for plotting
        keys = sorted(global_avg_mpjpe_gcn.keys())

        # Define the bar width and positions
        bar_width = 0.35
        indices = np.arange(len(keys))

        # Plotting bars for GCN and VideoPose3D
        plt.bar(indices , [global_avg_mpjpe_vp3d[key] for key in keys], bar_width, label='VideoPose3D-Real', alpha=0.8)
        plt.bar(indices + bar_width, [global_avg_mpjpe_gcn[key] for key in keys], bar_width, label='GCN-Specific', alpha=0.8)


        plt.title('Average MPJPE vs. Occlusion GT',  fontsize=20)
        plt.xlabel('Number of occluded keypoints',  fontsize=20)
        plt.ylabel('Average MPJPE [mm]',  fontsize=20)
        #plt.xticks(indices + bar_width / 2, [f"{key:.1f}" for key in keys],  fontsize=20)  # Positioning the x-axis labels
        plt.xticks(indices + bar_width / 2, [round(key*17/100) for key in keys],  fontsize=20)
        plt.yticks(fontsize=24)
        plt.grid(axis='y')
        plt.legend(fontsize=18)
        plt.tight_layout()
        # Save the plot to a file
        plt.savefig("avg_mpjpe_vs_occlusion_barplot.png")

        # Show the plot
        plt.show()
        
    if args.plot_type=="comparison":
        # Provided results organized into dictionaries
        
        videopose3d_results = {
            "Purchases": 103.85,
            "WalkDog": 103.77,
            "Posing": 90.93,
            "Greeting": 92.62,
            "Eating": 93.56,
            "Waiting": 94.66,
            "Discussion": 98.88,
            "Photo": 130.99,
            "Directions": 83.51,
            "Phoning": 110.14,
            "Sitting": 126.10,
            "Walking": 83.05,
            "WalkTogether": 91.65,
            "SittingDown": 188.27,
            "Smoking": 98.59
        }

        gcn_results = {
            "Directions": 68.51,
            "Discussion": 83.48,
            "Eating": 79.43,
            "Greeting": 76.32,
            "Phoning": 97.59,
            "Photo": 113.54,
            "Posing": 77.95,
            "Purchases": 88.76,
            "Sitting": 114.83,
            "SittingDown": 182.01,
            "Smoking": 85.79,
            "Waiting": 79.76,
            "WalkDog": 90.13,
            "Walking": 69.22,
            "WalkTogether": 74.54
        }
        
        '''
        videopose3d_results = {
            "Phone": 106.46,
            "PhonePacing": 106.15,
            "WalkCircle": 71.92,
            "ShoppingBag": 85.24,
            "Thinking": 84.61,
            "Directions": 72.97,
            "Arguing": 86.10
        }

        gcn_results = {
            "WalkCircle": 60.85,
            "Phone": 101.53,
            "Thinking": 78.49,
            "PhonePacing": 111.75,
            "Arguing": 73.71,
            "ShoppingBag": 76.77,
            "Directions": 62.40
           
        }
        '''

        # Create the bar plot
        plt.figure(figsize=(15, 8))

        actions = sorted(videopose3d_results.keys())  # Using sorted actions for ordering
        indices = np.arange(len(actions))
        bar_width = 0.35

        # Plotting bars for VideoPose3D and GCN
        plt.bar(indices, [videopose3d_results[action] for action in actions], bar_width, label='VideoPose3D', alpha=0.8)
        plt.bar(indices + bar_width, [gcn_results[action] for action in actions], bar_width, label='GCN', alpha=0.8)

        plt.title('MPJPE Comparison: VideoPose3D vs. GCN', fontsize=20)
        plt.xlabel('Actions', fontsize=20)
        plt.ylabel('MPJPE (mm)', fontsize=20)
        plt.xticks(indices + bar_width / 2, actions, rotation=45, ha='right', fontsize=20)  # Positioning the x-axis labels
        plt.yticks(fontsize=20)
        plt.grid(axis='y')
        plt.legend(fontsize=16)
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig("videopose3d_vs_gcn_comparison.png")

        # Show the plot
        plt.show()
        
    if args.plot_type=="comparison-avg":
        # Extracting average values from the dictionaries
        # New average values for the models tested on S9
        videopose3d_avg_s9 = 106.0
        gcn_avg_s9 = 92.12

        videopose3d_avg_s52 =   87.6
        gcn_avg_s52 = 80.78

        # Creating the bar plot for average values
        plt.figure(figsize=(10, 6))

        # Defining the labels
        labels = ['VideoPose3D S9', 'GCN S9','VideoPose3D S52', 'GCN S52']
        positions = np.arange(len(labels))
        values = [videopose3d_avg_s9, gcn_avg_s9, videopose3d_avg_s52, gcn_avg_s52]


        # Plotting bars for average values
        plt.bar(positions, values, alpha=0.8)

        # Adding title and labels
        plt.title('Average MPJPE Comparison: VideoPose3D vs. GCN', fontsize=20)
        plt.ylabel('MPJPE (mm)', fontsize=20)
        plt.xticks(positions, labels, fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(axis='y')
        plt.tight_layout()

        # Save the plot to a file
        plt.savefig("combined_average_comparison.png")

        # Show the plot
        plt.show()
        
    

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)