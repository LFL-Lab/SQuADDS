from setuptools import setup, find_packages

# Read requirements from the requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='SQuADDS',
    version='0.0.1',
    packages=find_packages(),
    description='Our project introduces an open-source database of programmatically generated and experimentally validated superconducting quantum device designs, accessible through a user-friendly interface, significantly lowering the entry barrier for research in this field.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sadman Ahmed Shanto',
    author_email='shanto@usc.edu',
    url='https://github.com/LFL-Lab/SQuADDS',
#    install_requires=required, 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
