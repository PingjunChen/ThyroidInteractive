Interactive Thyroid Whole Slide Image Diagnostic System using Deep Representation
----------------

### Intro
With the aim of computer-aided diagnosis and the insights that suspicious regions are generally easy to identify,
we develop an interactive histopathology whole slide image diagnostic system based on the suspicious regions
preselected by pathologists. We propose to generate the feature representation for these suspicious regions
via extracting and fusing patch features using deep neural networks. The pipeline of the proposed system is shown as
below:

![roi_wsi_system](roi_wsi_system.png)


### Methods
#### 1.1 patchCLS
- Patch classifier for feature extraction.

#### 1.2 genFeas
- Generate the representation for these suspicious regions for following analysis.

#### 1.3 roiCLS
- Diagnose the suspicious region

#### 1.4 retrieval
- Retrieve similar regions for reference
