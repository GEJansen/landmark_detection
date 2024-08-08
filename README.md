
# Landmark Detection in Medical Images

## Project Overview

This project implements the methods described in the paper [“Deep Learning-Based Regression and Classification for Automatic Landmark Localization in Medical Images”](https://ieeexplore.ieee.org/abstract/document/9139480). The method employs a global-to-local localization approach using fully convolutional neural networks (FCNNs) for accurate anatomical landmark detection in medical images.

## Method

### Overview
The method employs a two-step approach:
1. **Global Localization**: A global FCNN localizes multiple landmarks by analyzing image patches. It performs both regression (to determine displacement vectors) and classification (to predict the presence of landmarks in patches).
2. **Local Refinement**: Specialized FCNNs refine the globally localized landmarks by further analyzing local sub-images.

### Network Architecture
- **Global FCNN**: Based on ResNet34, modified to handle 3D images. It outputs displacement vectors and classification probabilities.
- **Local FCNN**: Similar but smaller network to refine the global landmark locations.

### Training and Inference
- **Training**: The networks are trained using a combination of mean absolute error for regression and binary cross-entropy for classification.
- **Inference**: During inference, the global FCNN predicts initial landmark locations, which are then refined by the local FCNNs.

### Datasets and Evaluation
The method was evaluated on three different datasets:
- **3D Coronary CT Angiography (CCTA)** (8 classes)
- **3D Olfactory bulb in MR** (1 class)
- **2D Cephalometric X-rays** (19 classes)

Key results include:
- For CCTA, the method achieved median Euclidean distance errors ranging from 1.45 to 2.48 mm.
- For olfactory MR, the median distance error was 0.90 mm.
- For cephalometric X-rays, errors ranged from 0.46 to 2.12 mm for different landmarks.

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd landmark_detection
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation
Modify the paths in the configuration files located in `experiment settings/` to make sure they match with your local dataset paths.

### Training

The experiments are divided into two main steps:
1. **First Step**: Initial training on datasets.
2. **Second Step**: Fine-tuning detection using a second (smaller) network for improved performance.

To train the models, navigate to the appropriate directory under `code/` and run the desired script. 
For example, to train the cephalometric model:
```bash
bash ./run_train_cephalometric.sh
```
Modify the configuration files in `experiment settings/` as needed to suit your experimental setup.
The experiment settings provided in that directory are the settings used in the paper.

## Inviting Contributions

The datasets used in the original paper are either private or no longer available. 
However, a new challenge dataset for cephalometric landmark detection is [available](https://cl-detection2023.grand-challenge.org/). 
We invite the community to adapt the existing code to accommodate this new dataset, or contribute by adapting the code to other datasets.

## References

If you find this repository useful to your work, please cite the original paper ([Deep Learning-Based Regression and Classification for Automatic Landmark Localization in Medical Images](https://ieeexplore.ieee.org/abstract/document/9139480)):
```
@article{noothout2020deep,
  title={Deep learning-based regression and classification for automatic landmark localization in medical images},
  author={Noothout, Julia MH and De Vos, Bob D and Wolterink, Jelmer M and Postma, Elbrich M and Smeets, Paul AM and Takx, Richard AP and Leiner, Tim and Viergever, Max A and I{\v{s}}gum, Ivana},
  journal={IEEE transactions on medical imaging},
  volume={39},
  number={12},
  pages={4011--4022},
  year={2020},
  publisher={IEEE}
}
```
