<center>
  <img src="docs/_static/images/squadds_logo_dark_name.png" width="80%" alt="SQuADDS Logo" /> <!-- This will stretch the logo to the full container width -->
</center>

# ![Alpha Version](https://img.shields.io/badge/Status-Alpha%20Version-yellow) Superconducting Qubit And Device Design and Simulation database

> :warning: **This project is an alpha release and currently under active development. Some features and documentation may be incomplete. Please update to the latest release.**

The SQuADDS (Superconducting Qubit And Device Design and Simulation) Database Project is an open-source resource aimed at advancing research in superconducting quantum device designs. It provides a robust workflow for generating and simulating superconducting quantum device designs, facilitating the accurate prediction of Hamiltonian parameters across a wide range of design geometries.

**Paper Link:** [SQuADDS: A Database for Superconducting Quantum Device Design and Simulation](https://arxiv.org/pdf/2312.13483.pdf)

**Website Link:** [SQuADDS](https://lfl-lab.github.io/SQuADDS/)

## Table of Contents

- [Installation](#installation)
- [Tutorials](#tutorials)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Next Release:](#next-release)

---

## Setup

Install using pip:

```bash
pip install SQuADDS
```

Install from source:

### Installation

1. Clone the repository.

```bash
git clone https://github.com/LFL-Lab/SQuADDS.git
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Run the setup script locally

```bash
pip install -e .
```

## Tutorials

- [Tutorial 1: Getting Started with SQuADDS](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-1_Getting_Started_with_SQuADDS.html)
- [Tutorial 2: Simulating Interpolated Designs](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-2_Simulate_interpolated_designs.html)
- [Tutorial 3: Contributing to the SQuADDS Database](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-3_Contributing_to_SQuADDS.html)
- [(COMING SOON) Tutorial 4: Adding your own Qubit Hamiltonian Calculator]()
- [(COMING SOON) Tutorial 5: Creating your own Interpolater]()
- [(COMING SOON) Tutorial 6: Creating your own Simulator]()

## Citation

If you use SQuADDS in your research, please cite the following paper:

```bibtex
    @article{SQuADDS,
        title={SQuADDS: A validated design database and simulation workflow for superconducting qubit design},
        author={Sadman Ahmed Shanto, Andre Kuo, Clark Miyamoto, Haimeng Zhang, Vivek Maurya, Evangelos Vlachos, Malida Hecht, Chung Wa Shum and Eli Levenson-Falk},
        journal={arXiv preprint arXiv: https://arxiv.org/pdf/2312.13483.pdf},
        year={2023}
    }
```

## Contributing

Contributions are welcome! If you have improvements or additions to the database, please follow these steps:

- Fork the repository.
- Create a new branch for your feature.
- Add your contributions.
- Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For inquiries or support, please contact [Sadman Ahmed Shanto](mailto:shanto@usc.edu).

---

## Next Release:

### [Version 0.3 Features](next_release.md):

---
