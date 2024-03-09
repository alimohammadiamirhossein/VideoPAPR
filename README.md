# Project: VideoPAPR - Video Proximity Attention Point Rendering

## Parts to be Done:

### 1. Integration of Zero-1-to-3 and AnimateDiff Models

- Take the diffusion model from [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) and incorporate the LoRA weights from [AnimateDiff](https://github.com/guoyww/AnimateDiff).
- Fine-tune the combined model using the [Objaverse](https://objaverse.allenai.org/) dataset.

### 2. Modification of PaPR Framework

- Modify the [PaPR](https://github.com/zvict/papr) framework to share weights across frames and alter the point cloud.
- Implement a new loss function such as L2-Norm or score-jacobian, and introduce a regularizer to minimize motion.

### 3. Motion Modification Using Objaverse Data

- Refine the motion aspect by incorporating moving objects from the [Objaverse](https://objaverse.allenai.org/) dataset.
