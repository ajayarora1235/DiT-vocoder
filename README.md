# DiT-vocoder

This repository contains an implementation of the DiT-based neural network architecture for spectrogram estimation described in:

**"Enhancing Spectrogram Realism in Singing Voice Synthesis via Explicit Bandwidth Extension Prior to Vocoder"**  
*Runxuan Yang, Kai Li, Guo Chen, Xiaolin Hu*  
arXiv:2508.01796 [cs.SD] (August 2025)

## Paper Abstract

This paper addresses the challenge of enhancing the realism of vocoder-generated singing voice audio by mitigating the distinguishable disparities between synthetic and real-life recordings, particularly in high-frequency spectrogram components. The proposed approach combines two innovations: an explicit linear spectrogram estimation step using denoising diffusion process with DiT-based neural network architecture optimized for time-frequency data, and a redesigned vocoder based on Vocos specialized in handling large linear spectrograms with increased frequency bins.

## Implementation

This repository implements the DiT (Diffusion Transformer) component of the paper's approach, which is used for explicit linear spectrogram estimation through denoising diffusion processes. The architecture is specifically optimized for time-frequency data to enhance spectrogram realism in singing voice synthesis.

### Files

- `DiT.py` - Main DiT implementation
- `Linear_MSE.py` - Linear MSE loss implementation
- `transformer.py` - Transformer architecture components

## Citation

```bibtex
@article{yang2025enhancing,
  title={Enhancing Spectrogram Realism in Singing Voice Synthesis via Explicit Bandwidth Extension Prior to Vocoder},
  author={Yang, Runxuan and Li, Kai and Chen, Guo and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2508.01796},
  year={2025}
}
```

## Paper Link

[arXiv:2508.01796](https://arxiv.org/abs/2508.01796)