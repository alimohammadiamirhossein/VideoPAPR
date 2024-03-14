# Project: VideoPAPR - Video Proximity Attention Point Rendering

## Parts to be Done:

### 1. Integration of Zero-1-to-3 and AnimateDiff Models

- Take the diffusion model from [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) and incorporate the LoRA weights from [AnimateDiff](https://github.com/guoyww/AnimateDiff).
- Fine-tune the combined model using the [Objaverse](https://objaverse.allenai.org/) dataset.

### 2. Modification of PaPR Framework

- Combine [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) and [PAPR](https://github.com/zvict/papr) to create a point cloud rendering model.
- Basically, we replace the 3D Reconstruction (SJC) part of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) with the [PAPR](https://github.com/zvict/papr).
- Implement a score-jacobian loss function, and introduce a regularizer to minimize motion.

### 3. Motion Modification Using Objaverse Data

- Modify the [PAPR](https://github.com/zvict/papr) framework to share weights across frames and alter the point cloud.
- Refine the motion aspect by incorporating moving objects from the [Objaverse](https://objaverse.allenai.org/) dataset.
