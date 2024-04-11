# Project: Generation of Animated 3D Characters

## Introduction:

The Generation of Animated 3D Characters project focuses on the development of advanced techniques for generating animated 3D characters in digital media. This project combines state-of-the-art methods in 3D modeling, animation, and rendering to create lifelike and dynamic characters for various applications in entertainment, gaming, virtual reality, and more.

## Parts to be Done:

### 1. Integration of Zero-1-to-3 and AnimateDiff Models

In this phase, we will integrate the diffusion model from [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) with the LoRA weights from [AnimateDiff](https://github.com/guoyww/AnimateDiff). This integration will allow us to generate high-quality animations with realistic motion and appearance. We will fine-tune the combined model using the [Objaverse](https://objaverse.allenai.org/) dataset, which includes diverse animation samples.

### 2. Modification of PaPR Framework

The next phase involves modifying the [PAPR](https://github.com/zvict/papr) framework to incorporate a point cloud rendering model for animated 3D characters. We will replace the 3D Reconstruction (SJC) part of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) with the [PAPR](https://github.com/zvict/papr) framework, enabling us to generate multi-view images for each frame of the animation. Additionally, we will implement advanced techniques for enhancing the realism and expressiveness of the characters.

### 3. Motion Modification Using Objaverse Data

In the final phase, we will further enhance the motion representation in the animated 3D characters by adapting the [PAPR](https://github.com/zvict/papr) framework to share weights across frames and modify the point cloud accordingly. By incorporating moving objects from the [Objaverse](https://objaverse.allenai.org/) dataset, we aim to create realistic and dynamic animations with lifelike movements and interactions. For usage instructions and setup details, please refer to the README.md in the phase3 folder. It is recommended to build a separate .venv for each phase to manage dependencies effectively.

## Conclusion:

The Generation of Animated 3D Characters project aims to advance the state of the art in character animation by leveraging cutting-edge techniques in 3D modeling, animation, and rendering. By integrating existing models and fine-tuning them with diverse datasets, we strive to create animated 3D characters that are both visually stunning and emotionally engaging. We believe that the outcomes of this project will have significant implications for various industries, including entertainment, gaming, virtual reality, and more.
