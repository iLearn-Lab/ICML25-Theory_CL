<div align="center">

# Understanding the Forgetting of (Replay-based) Continual Learning via Feature Learning: Angle Matters

**[Hongyi Wan](https://scholar.google.com/citations?hl=en&user=KHx7ReMAAAAJ)<sup>\*1</sup>**, **[Shiyuan Ren](https://openreview.net/profile?id=~Shiyuan_Ren1)<sup>\*1</sup>**, **[Wei Huang](https://weihuang05.github.io/)<sup>†2</sup>** **[Miao Zhang](https://miaozhang0525.github.io/)<sup>†1</sup>**, **[Xiang Deng](https://openreview.net/profile?id=~Xiang_Deng6)<sup>1</sup>**, **[Yixin Bao](https://openreview.net/profile?id=~Yixin_Bao1)<sup>1</sup>**, **[Liqiang Nie](https://liqiangnie.github.io/index.html)<sup>1</sup>** \
<sup>1</sup> Harbin Institute of Technology, Shenzhen \
<sup>2</sup> Independent Researcher \
\* Equal contribution \
† Corresponding author

---

[![ICML2025](https://img.shields.io/badge/ICML-2025-blue)
](https://img.shields.io/badge/ICML-2025-blue
)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Main Results](#-main-results)
- [Usage](#usage)
- [Citation](#citation)
- [Related Project](#related-project)
- [License](#-license)

---

## 📌 Introduction

Welcome to the official repository for **Feature Learning Theory in CL**. This project provides the main results of our ICML 2025 paper, presenting how the angle between task signal vectors influences forgetting.

*Disclaimer: These results are intended for research purposes.*

---

## 📊 Main Results

Acute and mildly obtuse angles lead to benign forgetting, whereas larger obtuse angles result in harmful forgetting. Moreover, a mid-angle sampling strategy outperforms existing sampling methods and can be seamlessly integrated into various replay-based frameworks.

---
## Usage

You can employ both [synthetic](.\synthetic) and [real-world](.\real-world) datasets to validate our theoretical results. In addition, the effectiveness of mid-angle sampling and EWC-Replay can be further examined through the [sampling-ewc](.\sampling-ewc).

---

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{wan2025understanding,
  title={Understanding the forgetting of (replay-based) continual learning via feature learning: Angle matters},
  author={Wan, Hongyi and Ren, Shiyuan and Huang, Wei and Zhang, Miao and Deng, Xiang and Bao, Yixin and Nie, Liqiang},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

---

## Related Project
[Mammoth - A PyTorch Framework for Benchmarking Continual Learning](https://github.com/aimagelab/mammoth)

---

## 📄 License

This project is released under the Apache License 2.0. See [`LICENSE`](./LICENSE) for details.
