# Image-Criterions-Python&MATLAB

This repository contains python implementation of SSIM，PSNR，NIQE，VIF

FSIM just can find in matlab，try to repreduce in python soon
[link](https://blog.csdn.net/ccheng_11/article/details/88554902)


## Dependencies
1) Python (>=3.5)
2) Numpy (>=1.16)
3) Python Imaging Library (PIL) (>=6.0)
4) Steerable Pyramid Toolbox (PyPyrTools) [link](https://github.com/LabForComputationalVision/pyPyrTools)

## Usage
Let imref and imdist denote reference and distorted images respectively. Then the VIF value is calculated as
VIF = vifvec(imref, imdist)

A demo code is provided in test.py for testing purposes

[1]H.R. Sheikh, A.C. Bovik and G. de Veciana, "An information fidelity criterion for image quality assessment using natural scene statistics," IEEE Transactions on Image Processing , vol.14, no.12pp. 2117- 2128, Dec. 2005.
