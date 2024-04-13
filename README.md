# GCN-Pose-Refinement
<p align="center">
  <img src="https://github.com/FilipaLino/GCN-Pose-Refinement/assets/102179022/8d6aafcc-fba8-4a8b-9134-0c0a32154d87" width="50%" height="50%">
</p>

This repository contains the implementation of a Graph Convolutional Network (GCN) designed for the refinement of estimated 3D poses. Our GCN model integrates spatial and temporal insights to enhance the accuracy of 3D human pose estimation, especially effective in scenarios with various types of occlusions. The model is trained using the [BlendMimic3D](https://github.com/FilipaLino/BlendMimic3D) dataset, which includes diverse occlusion scenarios, allowing the network to effectively learn and adapt to different occlusion types. For more detailed information about the project, interactive demos, and downloads related to the BlendMimic3D project, please visit our [project webpage](https://filipalino.github.io/filipalino.github.io-BlendMimic3D/).

## Model Description
The GCN model conceptualizes the human body as a graph structure with nodes representing body keypoints and edges denoting joint connections. It employs advanced graph convolutions that consider both spatial relationships and temporal continuity across frames. This method enables the model to accurately infer occluded or ambiguous keypoints in challenging environments.
<p align="center">
  <img src="https://github.com/FilipaLino/GCN-Pose-Refinement/assets/102179022/d9ec5c65-8c43-4790-b049-fd2a7228b6d9" width="30%" height="30%">
</p>


### Key Features:
- Incorporates both spatial and temporal graph convolutions.
- Utilizes a graph structure to capture complex joint interactions across frames.
- Refines pose estimations by addressing self-occlusions, object-based occlusions, and out-of-frame occlusions.
- Employs multi-class kernels for different types of neighboring nodes, enhancing the model's ability to generalize across various poses and conditions.


 ## Citation
If you use this model in your research, please cite our paper: 
```
[ citation format]
```

## Acknowledgements
This work was supported by LARSyS funding (DOI: 10.54499/LA/P/0083/2020, 10.54499/UIDP/50009/2020, and 10.54499/UIDB/50009/2020) and 10.54499/2022.07849.CEECIND/CP1713/CT0001, through Fundação para a Ciência e a Tecnologia, and by the SmartRetail project [PRR - C645440011-00000062], through IAPMEI - Agência para a Competitividade e Inovação.
<p align="center">
  <img src="https://github.com/FilipaLino/BlendMimic3D/assets/102179022/670897d0-1f7d-43e8-b63b-b2b961242730" width="50%" height="50%">
</p>
