from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='active_zero2',
        version='0.0.1',
        description="ActiveZero++",
        install_requires=["timm"],
        packages=find_packages(include=["active_zero2*"]),
    )
