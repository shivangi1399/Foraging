import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='convert_unreal_coordinates',
    version='0.0.1',
    description='Coordinate conversion for DomeVR',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://git.esi.local/havenith-scholvinck-lab/convert_unreal_coordinates',
    project_urls = {
        "Bug Tracker": "http://git.esi.local/havenith-scholvinck-lab/convert_unreal_coordinates/-/issues"
    },
    packages=['convert_unreal_coordinates'],
)
