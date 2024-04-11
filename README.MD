# Project: VideoPAPR - Video Proximity Attention Point Rendering

## Introduction:

VideoPAPR is a project aimed at enhancing video rendering capabilities by combining state-of-the-art techniques in diffusion models and point cloud rendering. This project leverages the strengths of existing models such as Zero-1-to-3 and PAPR to generate high-quality novel view videos with accurate motion representation.

## Parts to be Done:

### 1. Integration of Zero-1-to-3 and AnimateDiff Models

In this phase, we will integrate the diffusion model from [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) with the LoRA weights from [AnimateDiff](https://github.com/guoyww/AnimateDiff). This integration will allow us to combine spatial and temporal priors for video synthesis. We will fine-tune the combined model using the [Objaverse](https://objaverse.allenai.org/) dataset, which includes diverse animation samples.

### 2. Modification of PaPR Framework

The next phase involves modifying the [PAPR](https://github.com/zvict/papr) framework to incorporate a point cloud rendering model. We will replace the 3D Reconstruction (SJC) part of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) with the [PAPR](https://github.com/zvict/papr) framework, enabling us to generate multi-view images for each frame of the video. Additionally, we will implement a score-jacobian loss function and introduce a regularizer to minimize motion artifacts in the rendered videos.

### 3. Motion Modification Using Objaverse Data

In the final phase, we will further enhance the motion representation in the rendered videos by adapting the [PAPR](https://github.com/zvict/papr) framework to share weights across frames and modify the point cloud accordingly. By incorporating moving objects from the [Objaverse](https://objaverse.allenai.org/) dataset, we aim to create realistic and dynamic scenes in the rendered videos. For usage instructions and setup details, please refer to the README.md in the phase3 folder. It is recommended to build a separate .venv for each phase to manage dependencies effectively.

## Conclusion:

VideoPAPR project aims to push the boundaries of video rendering technology by combining cutting-edge techniques from diffusion models and point cloud rendering. By integrating existing models and fine-tuning them with diverse datasets, we strive to create novel view videos with high fidelity and accurate motion representation. We believe that the outcomes of this project will have significant implications for various applications in computer graphics and multimedia.
