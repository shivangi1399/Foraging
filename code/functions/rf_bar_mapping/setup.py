import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='rf',
    version='0.0.1',
    description='RF mapping analysis for DomeVR',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://git.esi.local/havenith-scholvinck-lab/rf_bar_mapping',
    project_urls = {
        "Bug Tracker": "http://git.esi.local/havenith-scholvinck-lab/rf_bar_mapping/-/issues"
    },
    packages=['rf'],
    install_requires=[
        # Github Repository
        'open-ephys-python-tools @ git+https://github.com/open-ephys/open-ephys-python-tools.git@0dd6d412264aefcd6169c3e2e009f5aca96314a8'
    ],
)
