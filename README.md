# ![Under Development](https://img.shields.io/badge/Status-Under%20Development-yellow) SQuADDS: 

> :warning: **This project is currently under active development. Features and documentation may be incomplete.**

The SQuADDS (Superconducting Qubit And Device Design and Simulation) Database Project is an open-source resource aimed at advancing research in superconducting quantum device designs. It provides a robust workflow for generating and simulating superconducting quantum device designs, facilitating the accurate prediction of Hamiltonian parameters across a wide range of design geometries.

Paper Link: [SQuADDS: A Database for Superconducting Quantum Device Design and Simulation](https://)
Website Link: [SQuADDS](https://sadmanahmedshanto.com/SQuADDS/)

## Table of Contents

- [Setup](#setup)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

--- 

## Setup

Install using pip:

```bash
pip install SQuADDS
```

Install from source:

### Prerequisites
- [Python 3.7+](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installing/)
- [Git](https://git-scm.com/downloads)

### Installation

1. Clone the repository.
```bash
git clone
```
2. Install the required packages.
```bash
pip install -r requirements.txt
```
3. Run the setup script.
```bash
python setup.py install
```

## Features

- **Data-Driven Interpolation**: Utilizes a comprehensive database for interpolating Hamiltonian parameters, ensuring high precision in predictions.
- **User-Specified Target Parameters**: Allows users to define target parameters such as qubit anharmonicity, coupling strength, resonator linewidth, and frequency.
- **Experimental Validation**: Includes experimentally measured data for enhancing the reliability and accuracy of the simulations.
- **Open-Source Collaboration**: Encourages contributions from the community, expanding the database and refining the simulation models.

## Tutorials
- [Tutorial 1: Getting Started with SQuADDS](https://sadmanahmedshanto.com/SQuADDS/tutorials/Tutorial-1_getting_started_with_SQuADDS.html)
- [Tutorial 2: Contributing to the SQuADDS Database](https://sadmanahmedshanto.com/SQuADDS/tutorials/Tutorial-2_Contributing_to_SQuADDS.html)


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

- [ ] More tutorials/examples/explanations on the website
- [ ] Contribution via HuggingFace Hub API
- [ ] More data points to existing configurations 
- [ ] Contribute data points to new configurations 

---