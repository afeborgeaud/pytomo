from setuptools import setup, find_packages

if __name__ == '__main__':
    with open('README.md', 'r') as f:
        long_description = f.read()

    setup(
        name="pytomo",
        version="1.0a0",
        author="Anselme Borgeaud",
        author_email="aborgeaud@gmail.com",
        license="MIT",
        description="Python tools for waveform inversion",
        long_description=long_description,
        url="https://github.com/afeborgeaud/pytomo",
        packages=find_packages(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            'obspy',
            'mpi4py',
            'pandas',
            'matplotlib',
            'geographiclib',
            'dsmpy',
            'numpy',
        ],
        python_requires='>=3.7',
        package_data={

        },
    )
