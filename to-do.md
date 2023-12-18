# PyPi Launch:

### 6. Build Your Package
- [ ] Use `setuptools` to create a distribution package.
- [ ] Run the command `python setup.py sdist bdist_wheel`.
- [ ] Check if the `dist` directory is created with your package files.

### 7. Install twine
- [ ] Install `twine` utility for publishing Python packages on PyPI.
- [ ] Run the command `pip install twine`.

### 8. Upload Your Package to PyPI
- [ ] Use `twine` to upload your package.
- [ ] Run the command `twine upload dist/*`.

### 9. Install Your Package
- [ ] Install your package using pip.
- [ ] Run the command `pip install your_package_name`.

### Notes
- Replace `your_package_name`, `your.email@example.com`, etc., with your actual package name and details.
- Ensure all dependencies your package needs are listed in `install_requires` in `setup.py`.
- Keep your package version updated in `setup.py`.

# Dev:

