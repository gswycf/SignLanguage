# An simple implementation based on SignGraph and MixSignGraph


Code and All files are in  "https://huggingface.co/hulala/SLRTmodel/tree/main"

## Data Preparation
 
1. PHOENIX2014 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). 

2. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

3. CSL dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

 
Download datasets and extract them, no further data preprocessing needed. 
 

### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{gan2024signgraph,
  title={SignGraph: A Sign Sequence is Worth Graphs of Nodes},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Wen, Hongkai and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13470--13479},
  year={2024}
}

@inproceedings{gan2023towards,
  title={Towards Real-Time Sign Language Recognition and Translation on Edge Devices},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  # SignSLRT-Baseline

  [![Hugging Face](https://img.shields.io/badge/huggingface-model-orange)](https://huggingface.co/hulala/SLRTmodel/tree/main)

  A compact baseline implementation built on ideas from SignGraph and MixSignGraph. This folder contains a lightweight reference for training and inference experiments and links to a ready-to-use model hosted on Hugging Face.

  This README shows how to get started quickly, where to find the code & model, and how to prepare standard datasets used in the experiments.

  ## Table of Contents

  - [Quick links](#quick-links)
  - [Quick start](#quick-start)
  - [Data preparation](#data-preparation)
  - [Usage examples](#usage-examples)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [Contact](#contact)

  ## Quick links

  - Hugging Face model & files: https://huggingface.co/hulala/SLRTmodel/tree/main
  - Parent project: see the repository root for `SignGraph/` and `MixSignGraph/` implementations

  ## Quick start

  1. Clone this repository and install dependencies (use the project's `requirements.txt` or your preferred environment manager):

  ```powershell
  git clone https://github.com/gswycf/SignLanguage.git
  cd SignLanguage
  python -m pip install -r SignGraph/requirements.txt
  ```

  2. Download or link a pretrained model from the Hugging Face page above if you want inference-only.

  3. Prepare datasets (see next section), then run the relevant script from the parent folders (this baseline reuses common training & eval entrypoints):

  ```powershell
  # Example: run evaluation with a checkpoint
  python SignGraph/main.py --config SignGraph/configs/phoenix2014.yaml --eval True --checkpoint ./path/to/checkpoint.pth
  ```

  Notes:

  - This folder focuses on a simple baseline and pointers to the model. For full training recipes, configs and distributed examples, check the `SignGraph/` and `MixSignGraph/` folders.

  ## Data preparation

  Supported datasets and where to obtain them:

  1. PHOENIX2014
     - RWTH-PHOENIX-Weather 2014: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

  2. PHOENIX2014-T
     - RWTH-PHOENIX-Weather 2014-T: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

  3. CSL (Chinese Sign Language)
     - Request / download: https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html

  After downloading, extract the datasets and point the dataset root(s) in the YAML configs under `SignGraph/configs/` or `MixSignGraph/configs/`. No additional preprocessing is required beyond the standard dataset layouts used by these benchmarks.

  ## Usage examples

  - Inference with a Hugging Face checkpoint

    1. Download model files from the Hugging Face link above.
    2. Run the parent inference script and point it to the downloaded checkpoint.

  ```powershell
  python SignGraph/main.py --config SignGraph/configs/phoenix2014.yaml --eval True --checkpoint ./checkpoints/hf_checkpoint.pth
  ```

  - Running a short train run (for debugging / development)

  ```powershell
  python MixSignGraph/main2.py --config MixSignGraph/configs/phoenix2014.yaml --epochs 1 --batch_size 4
  ```

  Adjust paths and flags in the YAML config files; they control dataset roots, save directories and logging.

  ## Citation

  If you use this baseline or the models, please consider citing the related work:

  ```latex
  @inproceedings{gan2025mixsigngraph,
    title={MixSignGraph: A Sign Sequence is Worth Mixed Graphs of Nodes},
    author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xie, Lei and Lu, Sanglu and Wen, Hongkai},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
  }

  @inproceedings{gan2024signgraph,
    title={SignGraph: A Sign Sequence is Worth Graphs of Nodes},
    author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Wen, Hongkai and Xie, Lei and Lu, Sanglu},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={13470--13479},
    year={2024}
  }

  @inproceedings{gan2023towards,
    title={Towards Real-Time Sign Language Recognition and Translation on Edge Devices},
    author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xie, Lei and Lu, Sanglu},
    booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
    pages={4502--4512},
    year={2023}
  }

  @inproceedings{gan2023contrastive,
    title={Contrastive learning for sign language recognition and translation},
    author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xia, Kang and Xie, Lei and Lu, Sanglu},
    booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23},
    pages={763--772},
    year={2023}
  }

  @article{han2022vision,
    title={Vision gnn: An image is worth graph of nodes},
    author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
    journal={Advances in neural information processing systems},
    volume={35},
    pages={8291--8303},
    year={2022}
  }
  ```

  ## Contributing

  Small fixes, documentation improvements and reproduction notes are welcome. For larger code changes or new model additions, please open an issue to start a discussion.

  Suggested workflow:

  1. Fork the repository and create a feature branch.
  2. Add tests or a short example demonstrating the change.
  3. Open a pull request with a clear description and any relevant logs.

  ## Contact

  - Maintainer: gswycf (GitHub)
  - For questions about the hosted model on Hugging Face, check the model card and discussion on the Hugging Face page.

  ---

  If you'd like, I can also add:

  - a small example notebook that downloads the Hugging Face model and runs a single inference, or
  - an environment file (`environment.yml`) for reproducibility.

  Tell me which you'd prefer and I'll add it.
