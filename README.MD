# Project: Generation of Animated 3D Characters

## Introduction:

The Generation of Animated 3D Characters project focuses on the development of advanced techniques for generating animated 3D characters in digital media. This project combines state-of-the-art methods in 3D modeling, animation, and rendering to create lifelike and dynamic characters for various applications in entertainment, gaming, virtual reality, and more.

<p float="left">
<img src="https://github.com/alimohammadiamirhossein/VideoPAPR/blob/main/phase1/data/gt_video-ezgif.com-video-to-gif-converter%20(1).gif" style="width: 350px; height: auto;">
<img src="https://github.com/alimohammadiamirhossein/VideoPAPR/blob/main/phase1/data/gt_video-ezgif.com-video-to-gif-converter%20(2).gif" style="width: 350px; height: auto;">
<img src="https://github.com/alimohammadiamirhossein/VideoPAPR/blob/main/phase1/data/gt_video-ezgif.com-video-to-gif-converter%20(3).gif" style="width: 350px; height: auto;">
<img src="https://github.com/alimohammadiamirhossein/VideoPAPR/blob/main/phase1/data/gt_video-ezgif.com-video-to-gif-converter.gif" style="width: 350px; height: auto;">
</p>

## Method:


### 1. Integration of Zero-1-to-3 and AnimateDiff Models

In the initial phase, we successfully integrated the diffusion model from [Zero123++](https://github.com/cvlab-columbia/zero123) with Stable Video Diffusion (SVD) using LoRA. We hope this integration allows us to generate high-quality animations with realistic motion and appearance. We fine-tuned the combined model using the [Objaverse](https://objaverse.allenai.org/) dataset, achieving promising results.

### 2. Modification of PaPR Framework

In the second phase, our focus shifted towards integrating the Score Jacobian Chaining technique into the [PAPR](https://github.com/zvict/papr) network. This integration aimed to enhance the performance and robustness of the PAPR network by leveraging score-based generative modeling techniques. For Score Jacobian Chaining, we used the 3D Reconstruction (SJC) part of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123).

### 3. Motion Modification Using Objaverse Data

we expanded the [PAPR](https://github.com/zvict/papr) framework to Animate-PAPR, enabling the generation of multi-view images for each frame of the video. This involved training the 3D asset features and the renderer of the [PAPR](https://github.com/zvict/papr) for the initial frame, followed by a freezing and initialization strategy to propagate features across consecutive frames. This iterative process enhanced the network's capacity to produce consistent multi-view renderings throughout the entire video sequence. By incorporating moving objects from the [Objaverse](https://objaverse.allenai.org/) dataset, we created realistic and dynamic animations with lifelike movements and interactions.

## Conclusion:

The Generation of Animated 3D Characters project has achieved significant milestones in advancing the state of the art in character animation. Through the successful completion of each phase, we can demonstrat the effectiveness of our approach in generating lifelike and dynamic animated characters. The outcomes of this project have the potential to revolutionize various industries, including entertainment, gaming, virtual reality, and more.
