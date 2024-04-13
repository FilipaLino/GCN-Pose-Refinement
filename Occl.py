import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


subject = 'S11'

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

def main():
    # Specify the directory you want to list files from
    directory_path = f"../Synthetic_H3.6M/{subject}/Occlusions"
    # List all filenames in the directory
    actions = os.listdir(directory_path)

    # Load the .npz GCN file
    gcn_spe = np.load(f'S{subject}_MPJPE/mpjpe_per_frame_Spe.npz', allow_pickle=True)
    gcn_gen = np.load(f'S{subject}_MPJPE/mpjpe_per_frame_Gen.npz', allow_pickle=True)

    # Load the .npz VideoPose3D file
    vp3d_real = np.load(f'../VideoPose3D/{subject}_MPJPE/mpjpe_per_frame_real.npz', allow_pickle=True)
    vp3d_merged = np.load(f'../VideoPose3D/{subject}_MPJPE/mpjpe_per_frame_merged.npz', allow_pickle=True)

    occl={}
    # Print all filenames
    for action in actions:
        aux=[]

        occlusions = np.load(os.path.join(directory_path, f"{action}/occluded_kpt.npz"), allow_pickle=True)
        for i in range(4):
            aux.append(occlusions[f'Cam_{i}'])
           
        occl[action] = np.concatenate(aux, axis=0)
    

    # Compute the occlusion percentages and average MPJPE values grouped by occlusion percentages for all actions
    results = {}
    # Aggregate MPJPE values and occlusion percentages across all actions
    all_occlusion_perc = []
    all_mpjpe_gcn = []
    all_mpjpe_vp3d = []
    vp3d=gcn_spe
    gcn=gcn_gen

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
    print('GCN-Specific', [global_avg_mpjpe_vp3d[key] for key in keys])
    print('GCN-General',[global_avg_mpjpe_gcn[key] for key in keys])

    # Plotting bars for GCN and VideoPose3D
    plt.bar(indices , [global_avg_mpjpe_vp3d[key] for key in keys], bar_width, label='VideoPose3D-Real', alpha=0.8)
    plt.bar(indices + bar_width, [global_avg_mpjpe_gcn[key] for key in keys], bar_width, label='VideoPose3D-Merged', alpha=0.8)


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

 
        
    

if __name__ == "__main__":
    main()