# hzero (EN)

[Leer en español](docs/README.es.md)

**hzero** is a statistical inference library focused on the implementation of hypothesis tests, both parametric and non-parametric. It is designed as an educational and experimental tool to explore the behavior of different statistical tests under various conditions.

## 🚧 Project Status

This proyect is currently under active development. Expect breaking changes and incomplete features.

🛠️ Currently working on the non-parametric tests. See [`feature/non-parametric`](https://github.com/eduvicgar/hzero/tree/feature/non-parametrical)

## ✨ Main Features

- Implementation of distributions used in hypothesis testing (such as Student's t, chi-square, etc.)  
- Support for parametric hypothesis tests (mean, variance, difference of means and variances ratio)
- Visualization of results with **matplotlib**  
- Extensive use of **NumPy** for efficient numerical operations  

> [!WARNING]
> Most of the classes provided aren't thoroughly tested, so it's possible to get unexpected results in edge cases.
> These bugs will be fixed throughout the development of the library. You may also submit an issue if you encounter a bug.

## 📦 Technologies Used

- [SciPy](https://scipy.org/)  
- [NumPy](https://numpy.org/)  
- [matplotlib](https://matplotlib.org/)  

## 🔧 Installation

> [!NOTE]  
> Currently, there is no standard way to install hzero. To use it, you can clone this repository:

```bash
git clone https://github.com/eduvicgar/hzero.git  
cd hzero
```

Then you can import the modules directly from your local development environment.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

## ✍️ Author

eduvicgar

## 📌 Upcoming Improvements
- Non-parametric hypothesis tests such as Pearson’s chi-square and Kolmogorov-Smirnov  
- Additional unit tests  
- Additional documentation and example notebooks  
