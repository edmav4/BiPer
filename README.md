<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[//]: # ([![Contributors][contributors-shield]][contributors-url])
[//]: # ([![Forks][forks-shield]][forks-url])

[//]: # ([![Stargazers][stars-shield]][stars-url])

[//]: # ([![Issues][issues-shield]][issues-url])

[//]: # ([![MIT License][license-shield]][license-url])
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/edmav4/BiPer">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">BiPer</h3>

  <p align="center">
    Binary Neural Networks using a Periodic Function
    <br />
    <a href="https://arxiv.org/pdf/2404.01278.pdf">Paper</a>
    ·
    <a href="">Supplementary Material</a>
    ·
    <a href="">Video</a>
  </p>
</div>


## News and Updates :newspaper_roll:

**April 1, 2024**
- Our preprint is available at [Arxiv](https://arxiv.org/abs/2404.01278)

**February 27, 2024**
- Our paper have been accepted to CVPR 2024!

<br />

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#clone-this-repository">Clone this repository</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#download-datasets">Download Datasets</a></li>
      </ul>
    </li>
    <li><a href="#cifar10">CIFAR10</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#results">Results</a></li>
      </ul>
    </li>
    <li><a href="#imagenet">ImageNet</a>
      <ul>
        <li><a href="#inet-train">Training</a></li>
        <li><a href="#inet-eval">Evaluation</a></li>
        <li><a href="#inet-res">Results</a></li>
      </ul>
    </li>
    <li><a href="#how-to-cite">How to Cite</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Abstract

Quantized neural networks employ reduced precision representations for both weights and activations. This quantization process significantly reduces the memory requirements and computational complexity of the network. Binary Neural Networks (BNNs) are the extreme quantization case, representing values with just one bit. Since the sign function is typically used to map real values to binary values, smooth approximations are introduced to mimic the gradients during error backpropagation. Thus, the mismatch between the forward and backward models corrupts the direction of the gradient causing training inconsistency problems and performance degradation. In contrast to current BNN approaches, we propose to employ a binary periodic (BiPer) function during binarization. Specifically, we use a square wave for the forward pass to obtain the binary values and employ the trigonometric sine function with the same period of the square wave as a differentiable surrogate during the backward pass. We demonstrate that this approach can control the quantization error by using the frequency of the periodic function and improves network performance. Extensive experiments validate the effectiveness of BiPer in benchmark datasets and network architectures, with improvements of up to 1% and 0.69% with respect to state-of-the-art methods in the classification task over CIFAR-10 and ImageNet, respectively. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Clone this repository
Clone our repo to your local machine using the following command:
```bash
git clone https://github.com/edmav4/BiPer.git
cd BiPer
```

### Prerequisites

Create a new conda environment using the provided `environment.yml` file.
  ```bash
  conda env create --prefix ./venv -f environment.yml
  conda activate ./venv
  ```

### Download Datasets
Our BiPer was trained on CIFAR-10 and ImageNet datasets. You can download the datasets using the following commands:

- CIFAR-10
  ```python
  python cifar10/dataset/download.py --dataset cifar10 --data_path cifar10/data/CIFAR10
  ```
- ImageNet
  
    See [ImageNet](imagenet/dataset/README.md) for more details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## CIFAR10

### Training
Our approach consists of a two-stage training strategy. In the first stage, the network is trained with real weights and binary features. Then, in the second stage, a warm weight initialization is employed based on the binary representation of the output weights from the first stage, and the model is fully trained to binarize the weights. Thus, the problem is split into two subproblems: weight and feature binarization.

#### Stage 1
To train stage1, you can use a similar command as follows:
```bash
# Example for BiPer-ResNet18 model
python -u main.py \
--gpus 0 \
--model resnet18_1w1a \
--results_dir ./result/stage1 \
--dataset cifar10 \
--epochs 600 \
--lr 0.021 \
-b 256 \
-bt 128 \
--lr_type cos \
--warm_up \
--weight_decay 0.0016 \
--tau 0.037 \
--freq 20
```
See this example in `run_stage1.sh`, and run it with `bash run_stage1.sh`.

#### Stage 2
After training the first stage, you can train the second stage using the following command:

```sh
# Example for BiPer-ResNet18 model
python -u main_stage2.py \
--gpus 0 \
--model resnet18_1w1a \
--results_dir ./result/stage2 \
--dataset cifar10 \
--epochs 300 \
--lr 0.0037 \
-b 256 \
-bt 128 \
--lr_type cos \
--warm_up \
--weight_decay 0.00016 \
--tau 0.0468 \
--load_ckpt_stage1 ./result/stage1/model_best.pth.tar
```
Note that `--load_ckpt_stage1` should be specified to load the pretrained model from the first stage. See this example in `run_stage2.sh`, and run it with `bash run_stage2.sh`.
### Evaluation

To evaluate a pretrained model, you can use the following command:

```sh
# see eval.sh
python main_stage2.py \
--gpus 0 \
-e {checkpoint_path} \
--model {model arch} \
--dataset cifar10 \
-bt 128 \
```
for example, using the pretrained model of BiPer-ResNet18:
```sh
# example ResNet18
python main_stage2.py \
--gpus 0 \
-e ./pretrained_models/biper_cifar10_resnet18_stage2/model_best.pth.tar \
--model resnet18_1w1a \
--dataset cifar10 \
-bt 128 \
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results

### Quantization Error
To compute the quantization error, you can use the following command:
```bash
python compute_QE.py
```
Please specify the model and data path in the script.

### Pretrained Models

|      Model      |   Dataset    | Params (M)  | Top-1  |                                               Config                                               |                                            Download                                            |
|:---------------:|:------------:|:-----------:|:------:|:--------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------:|
| BiPer-ResNet18  |   CIFAR-10   |    11.01    | 93.75  | [Config File](https://drive.google.com/file/d/1xyAyEmiiahOiMrbZt_8Ee8WStHW951tC/view?usp=sharing)  | [Model](https://drive.google.com/file/d/1vye9pCvRcfNtOrErZDtU5OKzA9_2-S_-/view?usp=drive_link) |
| BiPer-ResNet20  |   CIFAR-10   |    0.27     | 87.50  | [Config File](https://drive.google.com/file/d/14AE4O27soD8H7kdBpC40MpVBycykJS-N/view?usp=sharing)  | [Model](https://drive.google.com/file/d/1s5iMXNzY4jQmum-pjRX7AgFJ4Qcj9zQ5/view?usp=drive_link) |
| BiPer-VGG-Small |   CIFAR-10   |    4.66     | 92.460 | [Config File](https://drive.google.com/file/d/1nZDhab8SIbLyIqiOCvAek0lnMwb8LM_q/view?usp=sharing)  |[Model](https://drive.google.com/file/d/1mGFhfCENjYcqx-WCnqgBlq5sOToWfE-x/view?usp=drive_link)  |

[//]: # (\| [Log]&#40;URL_to_log&#41;)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## ImageNet

### Training
<a name="inet-train"></a>
Similar to CIFAR10, here we specify the training process for ImageNet.

#### Stage 1
To train stage1, you can use a similar command as follows:
```sh
# example BiPer-ResNet18
python main.py \
--gpus 0,1,2,3 \
--model resnet18_1w1a \
--data_path data \
--dataset imagenet \
--epochs 200 \
--lr 0.1 \
--weight_decay 1e-4 \
-b 512 \
-bt 256 \
--lr_type cos \
--freq 20 \
--warm_up \
--tau_min 0.85  \
--tau_max 0.99  \
--print_freq 250 \
--use_dali
```
See this example in `run_stage1.sh`, and run it with `bash run_stage1.sh`.
#### Stage 2
After training the first stage, you can train the second stage using a similar command as following:
```sh
python main_stg2.py \
--gpus 0 \
--model resnet18_1w1a \
--data_path data \
--dataset imagenet \
--epochs 100 \
--lr 0.01 \
-b 512 \
-bt 256 \
--lr_type cos \
--weight_decay 1e-4 \
--tau_min 0.0  \
--tau_max 0.0  \
--freq 20 \
--load_ckpt_2tage ./result/stage1/model_best.pth.tar \
--use_dali \
# --resume
```
See this example in `run_stage2.sh`, and run it with `bash run_stage2.sh`.

### Evaluation
<a name="inet-eval"></a>
To evaluate a pretrained model, you can use the following command:

```sh
# see eval.sh
python main_stage2.py \
--gpus 0 \
-e {checkpoint_path} \
--model {model arch} \
--dataset imagenet \
-bt 256
```
for example, using the pretrained model of ResNet18:
```sh
# example BiPer-ResNet18
python main_stage2.py \
--gpus 0 \
-e pretrained_models/biper_imagenet_resnet18_stage2/model_best.pth.tar \
--model resnet18_1w1a \
--dataset imagenet \
-bt 256
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- RESULTS -->
## Results
<a name="inet-res"></a>
### Pretrained Models

|     Model      |  Dataset   | Params (M) | Top-1 |      Config       |                                          Download                                           |
|:--------------:|:----------:|:----------:|:-----:|:-----------------:|:-------------------------------------------------------------------------------------------:|
| BiPer-ResNet18 | ImageNet1K |   11.69    | 61.40 | [Config File](https://drive.google.com/file/d/1Kvc82rtf1-bWB-GsLjybSFPyEqfz0y0e/view?usp=sharing) | [Model](https://drive.google.com/file/d/1mUPcohyRk4_NLWXO8TMcIWRb606wMSlc/view?usp=sharing) |
| BiPer-ResNet34 | ImageNet1K |   21.81    | 65.73 | [Config File](https://drive.google.com/file/d/1GdUNclCntpsE2XnqlTafJSf0Si-luxqc/view?usp=sharing) | [Model](https://drive.google.com/file/d/10ACakHYVBlWeoNACBbNwvFoIjk-kTwBn/view?usp=sharing) |

<p align="right">(<a href="#readme-top">back to top</a>)</p>
<!-- 11694376 Resnet18 -->

<!-- How to Cite -->
## How to cite

If you use the code or models from this project in your research, please cite our work as follows:

```Latex
@article{vargas2024biper,
  title={BiPer: Binary Neural Networks using a Periodic Function},
  author={Vargas, Edwin and Correa, Claudia and Hinojosa, Carlos and Arguello, Henry},
  journal={arXiv preprint arXiv:2404.01278},
  year={2024}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Biper is distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

- Edwin Vargas
  - Linkedin: https://www.linkedin.com/in/edwin-vargas-80ab7873/
  - Twitter: [@edmav47](https://twitter.com/edmav47)
  - Email: email@example.com
  - Webpage: https://www.researchgate.net/profile/Edwin-Vargas-13

- Carlos Hinojosa
  - Linkedin: https://www.linkedin.com/in/phdcarloshinojosa/
  - Twitter: [@CarlosH_93](https://twitter.com/CarlosH_93)
  - Email: carlos.hinojosamontero@kaust.edu.sa
  - Webpage: https://carloshinojosa.me/

<!-- Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name) -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- Our code is based on the ReCU repository: https://github.com/z-hXu/ReCU. We thank the authors for making their code publicly available.
- This work was supported by the Vicerrectoría de Investigacion y Extensión of Universidad Industrial de Santander (UIS), Colombia under the research project **VIE-3735**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/edmav4/BiPer/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/edmav4/BiPer/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/edmav4/BiPer/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/edmav4/BiPer/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/edmav4/BiPer/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
