# Sparse-Adversarial-Attack

This repository provides a simple implementation of our recent work ["Sparse Adversarial Attack via Perturbation Factorization"](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670035.pdf), ECCV 2020.


### Dependencies
* Python 3.6
* Pytorch 0.4.0 (other versions may be also OK, but we didn't verify it)
* Other Python packages: numpy, time, PIL, skimage, json

### Demo 

The following demo can generate sparse adversarial perturbations by attacking a CNN model trained on CIFAR-10, using the proposed attack method. 
```
python main.py --attacked_model cifar_best.pth --img_file img0.png --target 1 --k 200	
```
* Inputs: `attacked_model` indicates the checkpoint to be attacked; `img_file` denotes the benign image; `target` is the target attack class; `k` represents the number of perturbed pixels.
* Outputs: The generated perturbation (saved as .npy) and the adversarial image (saved as .png file) will be saved in `./results`. 

### Citations

```
@inproceedings{sapf-ECCV2020,
  title={Sparse Adversarial Attack via Perturbation Factorization},
  author={Fan, Yanbo and Wu, Baoyuan and Li, Tuanhui and Zhang, Yong and Li, Mingyang and Li, Zhifeng and Yang, Yujiu},
  booktitle={European conference on computer vision},
  year={2020}
}
```
