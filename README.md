## Quick Links

- [Building and Testing](#building-and-testing)
- [Usage](#usage)
- [Performance](#performance)
- [Reference](#reference)

## Building and Testing

This project is implemented primarily in Python 3.6, with several dependencies listed below. We have tested the framework on Ubuntu 16.04.5 LTS with kernel 4.4.0, and it is expected to easily build and run under a regular Unix-like system.

### Dependencies

- [Python 3.7](https://www.python.org).
  Version 3.7.0 has been tested. Higher versions are expected be compatible with current implementation, while there may be syntax errors or conflicts under python 2.x.

- [PyTorch](https://pytorch.org). 

  Version 1.7.0 has been tested. You can find installation instructions [here](https://pytorch.org/get-started/locally/). Note that the GPU support is **ENCOURAGED** as it greatly boosts training efficiency.


- [Other Python modules](https://pypi.python.org). Some other Python module dependencies are listed in ```requirements.txt```, which can be easily installed with pip:

  ```bash
  pip install -r requirements.txt
  ```

## Reference
[1] R. Wang, Y. Zhang, L. Peng, G. Fortino and P. -H. Ho, "Time-Varying-Aware Network Traffic Prediction Via Deep Learning in IIoT," in IEEE Transactions on Industrial Informatics, vol. 18, no. 11, pp. 8129-8137, Nov. 2022, doi: 10.1109/TII.2022.3163558.

```
@ARTICLE{9745370,
  author={Wang, Ranran and Zhang, Yin and Peng, Limei and Fortino, Giancarlo and Ho, Pin-Han},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Time-Varying-Aware Network Traffic Prediction Via Deep Learning in IIoT}, 
  year={2022},
  volume={18},
  number={11},
  pages={8129-8137},
  doi={10.1109/TII.2022.3163558}}
```# NetworkTrafficPrediction