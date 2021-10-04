import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = 'modelbricks',
    version='0.1.0',
    author='Jimmy Su',
    author_email = 'jim83531@gmail.com',
    description = 'ML model brick',
    long_description = long_description,
    long_description_content__type='markdown',
    url = 'https://git.owdev.net/Jimmy/modelbricks.git',
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'tensorflow>=2.5.0'
    ]
    
)