# Mapping Brainwave Segments to Processing Parameters

This document outlines a machine learning task that maps 5-second multiband EEG segments to Processing parameters encoding texture and color characteristics extracted from artwork.

## Problem Definition

We model the task as a function \( f: A \rightarrow B \), where:

* \( A \) is the set of EEG samples. Each element \( a \in A \) is a matrix in \( \mathbb{R}^{8 \times 3200} \). The eight rows correspond to spectral bands (delta, theta, alpha, low-beta, high-beta, low-gamma, mid-gamma, high-gamma), and the 3200 columns represent samples from a 640 Hz acquisition device over five seconds.
* \( B \) is the set of parameter vectors that describe color and texture features of artworks in a format directly usable by Processing sketches (e.g., color palettes, Perlin noise parameters, brushstroke densities).

Each training example pairs an EEG segment with the Processing parameter vector derived from the artwork that elicited the recorded brain activity.

## Dataset Specification

1. **EEG Collection**
   * Sampling rate: 640 Hz, 8 spectral bands.
   * Segment length: 5 seconds (3200 samples per band).
   * Preprocessing: band-pass filtering per band, artifact removal (e.g., ICA for eye blinks), z-score normalization per band.

2. **Artwork Feature Extraction**
   * Use texture descriptors (e.g., Gabor filters, Gray-Level Co-occurrence Matrix) to capture structural patterns.
   * Extract color representations (e.g., Lab color histograms, dominant palette via k-means).
   * Convert features to Processing parameters, such as noise scales, stroke orientations, color arrays, and animation speeds.

3. **Dataset Structure**
   * Store EEG tensors as `.npy` files with shape `(8, 3200)`.
   * Store Processing parameter vectors as JSON files containing structured fields for color palettes, texture coefficients, and motion parameters.
   * Maintain metadata (participant ID, artwork ID, timestamp) in a central CSV or database.

## Model Architecture

A baseline architecture can combine temporal convolution and attention mechanisms:

1. **Spectral-Temporal Encoder**
   * Apply 1D convolutions across time within each band to extract temporal motifs.
   * Use depthwise separable convolutions to reduce parameter count.
   * Introduce residual blocks with batch normalization and GELU activation.

2. **Cross-Band Attention Layer**
   * Flatten convolution outputs and apply multi-head self-attention across bands to model inter-band relationships.
   * Include positional encodings tied to time indices.

3. **Latent Projection**
   * Global average pooling to obtain a fixed-size latent vector.
   * Fully connected layers with dropout to project into the Processing parameter space.

4. **Output Layer**
   * Use separate heads for color palette logits (softmax over palette entries) and continuous texture parameters (linear outputs with appropriate bounds via activation functions).

## Training Procedure

1. **Loss Functions**
   * Use mean squared error (MSE) for continuous texture parameters.
   * Apply cross-entropy for categorical color palette selection.
   * Combine losses with weighting to balance texture and color accuracy.

2. **Optimization**
   * AdamW optimizer with cosine learning rate scheduling.
   * Mini-batch size: 32 segments, gradient clipping to stabilize training.

3. **Regularization**
   * SpecAugment-style temporal masking to improve robustness.
   * Dropout in fully connected layers.
   * Mixup augmentation across EEG segments triggered by similar artworks.

4. **Validation**
   * Split data into training/validation/test by participant to avoid leakage.
   * Monitor parameter reconstruction error and downstream rendering similarity metrics (e.g., SSIM between rendered sketches and target textures).

## Evaluation and Deployment

1. **Rendering Consistency**
   * Generate Processing sketches using predicted parameters and compare with reference artworks via perceptual metrics (LPIPS, SSIM).

2. **User Study**
   * Conduct qualitative evaluations by presenting generated sketches alongside original artworks to assess perceptual similarity.

3. **Real-Time Adaptation**
   * For interactive installations, slide a 5-second window across live EEG streams and feed predictions to Processing in real time.

## Future Enhancements

* Integrate Variational Autoencoders (VAEs) to learn a latent texture space directly from artworks, mapping EEG segments into that latent space before decoding to Processing parameters.
* Explore contrastive learning by aligning EEG embeddings with vision transformer embeddings of the artworks.
* Incorporate feedback loops where Processing outputs update stimulus selection for closed-loop neuroaesthetic experiences.

