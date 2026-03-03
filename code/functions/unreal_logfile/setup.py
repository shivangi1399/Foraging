import setuptools

setuptools.setup(
    name='unreal_logfile',
    version='0.0.2',
    description='Logfile parsing for DomeVR',
    url='http://git.esi.local/havenith-scholvinck-lab/unreal_logfile',
    project_urls = {
        "Bug Tracker": "http://git.esi.local/havenith-scholvinck-lab/unreal_logfile/-/issues"
    },
    py_modules=["parse_logfile", "EndOfDayFinal"],
)
