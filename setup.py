from setuptools import find_packages, setup


def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read requirements from files
core_requirements = read_requirements('requirements-core.txt')
optional_requirements = read_requirements('requirements-optional.txt')

setup(
    name='SQuADDS',
    version='0.3.7',
    packages=find_packages(),
    description='Our project introduces an open-source database of programmatically generated and experimentally validated superconducting quantum device designs, accessible through a user-friendly interface, significantly lowering the entry barrier for research in this field.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sadman Ahmed Shanto',
    author_email='shanto@usc.edu',
    include_package_data=True,
    url='https://github.com/LFL-Lab/SQuADDS',
    install_requires=core_requirements,
    extras_require={
        'optional': optional_requirements,
        'all': core_requirements + optional_requirements
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.10',
)
