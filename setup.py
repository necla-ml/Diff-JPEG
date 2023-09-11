from setuptools import setup

setup(
    name="diff_jpeg",
    version="0.1",
    url="https://github.com/ChristophReich1996/Differentiable_JPEG",
    license="MIT License",
    author="Christoph Reich",
    author_email="ChristophReich@gmx.net",
    description="PyTorch Differentiable JPEG Reich et al. [WACV 2024].",
    packages=[
        "diff_jpeg",
    ],
    install_requires=["torch>=1.0.0", "opencv-python>=4.7.0.72"],
)
