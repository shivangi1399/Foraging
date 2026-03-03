import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='preprocessing',
    version='0.0.3',
    description='Preprocessing scripts for ZeroNoise lab',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://git.esi.local/havenith-scholvinck-lab/preprocessing',
    project_urls = {
        "Bug Tracker": "http://git.esi.local/havenith-scholvinck-lab/preprocessing/-/issues"
    },
    packages=['preprocessing'],
    install_requires=[
        # Github Private Repository
        'open-ephys-python-tools @ git+https://github.com/open-ephys/open-ephys-python-tools.git@0dd6d412264aefcd6169c3e2e009f5aca96314a8'
    ],
)
