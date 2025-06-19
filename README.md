# SCIANN: EEG-Based Empathy Recognition

This repository provides the PyTorch implementation for EEG-based empathy recognition tasks.

* **SCIANN was tested on the following datasets and achieved SOTA results**:
1.  A **self-collected dataset** from a two-person collaborative motor imagery experiment.
2.  The **public ECSU-PCE dataset** focused on social touch interactions.

## Tasks Supported

The codebase allows training and evaluation for the following classification tasks:

* **On Self-Collected Data:**
    * **Empathy Presence Detection:** Binary classification (e.g., distinguishing friend vs. stranger interactions).
    * **Empathy Type Recognition:** 4-class classification (distinguishing interaction conditions).

* **On ECSU-PCE Public Data:**
    * **Social Perception Classification:** Binary classification based on Perceptual Awareness Scale (PAS) scores (e.g., distinguishing high vs. low perceived presence/empathy).

## This model and all the codes in this repository are designed and developed independently by Haiyang Long. If you refer to or use the code of this study, please mention me in your project.

## The related article of this model was published in IEEE Journal of Biomedical and Health Informatics(JBHI), titled EEG-based Spatial-Channel Interaction Attention Neural Networks for Detecting Empathy in Motor Collaboration, which is currently in early access, doi: 10.1109/JBHI.2025.3580617

## Contact

For questions, please use GitHub Issues or contact hp375169@gmail.com.
