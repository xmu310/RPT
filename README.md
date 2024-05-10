## Faint Curved Edge Detection (RPT version)

This repository is for image processing course project, and contains a Python implementation of a faint edge detection algorithm for noisy images based on the principles described in the CVPR 2016 paper "Fast Detection of Curved Edges at Low SNR" by Ofir et al.

The original paper can be found at [https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Ofir_Fast_Detection_of_CVPR_2016_paper.html](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Ofir_Fast_Detection_of_CVPR_2016_paper.html).

The code repository is available at [https://github.com/NatiOfir/FaintCurvedEdgeDetection](https://github.com/NatiOfir/FaintCurvedEdgeDetection).

### Overview

The provided Python code offers a simplified version of the original MATLAB code, making it easier to understand and use for educational purposes and real-world applications. The algorithm is designed to detect curved edges in images with low signal-to-noise ratio (SNR), providing robust performance even in noisy conditions.

### Features

- Detection of faint curved edges in noisy images
- Implementation based on the CVPR 2016 paper
- Python code for ease of use and understanding
- Simplified compared to the original MATLAB code

### Usage

1. Clone the repository:

   ```
   git clone https://github.com/xmu310/RPT.git
   ```

2. Navigate to the cloned directory:

   ```
   cd RPT
   ```

3. Run the Python script to detect curved edges in your images:

   ```
   python3 demo.py Images/image_name
   ```

### Dependencies

This project relies on the following Python libraries:

- NumPy
- Scikit-image
- Matplotlib
- H5py

You can install these dependencies using pip:

```
pip install numpy scikit-image matplotlib h5py
```

These libraries are necessary for running the provided scripts and performing various image processing tasks.

### Acknowledgments

This project is inspired by the work of Ofir et al. and is developed as a part of an image processing course project.
