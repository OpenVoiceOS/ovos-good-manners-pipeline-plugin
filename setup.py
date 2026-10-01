#!/usr/bin/env python3
from setuptools import setup
from os import walk, path

BASEDIR = path.abspath(path.dirname(__file__))
URL = "https://github.com/OpenVoiceOS/ovos-good-manners-pipeline-plugin"
PLUGIN_CLAZZ = "GoodMannersPipeline"  # needs to match __init__.py class name
PYPI_NAME = "ovos-good-manners-pipeline-plugin"  # pip install PYPI_NAME
PLUGIN_PKG = PYPI_NAME.replace("-", "_")
# pipeline plugins register under 'opm.pipeline', keyed by the plugin's
# own name - this is the id that goes into intents.pipeline
PLUGIN_ENTRY_POINT = f"{PYPI_NAME}={PLUGIN_PKG}:{PLUGIN_CLAZZ}"


def get_version():
    """Find the version of the package"""
    version_file = path.join(BASEDIR, "version.py")
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if "VERSION_MAJOR" in line:
                major = line.split("=")[1].strip()
            elif "VERSION_MINOR" in line:
                minor = line.split("=")[1].strip()
            elif "VERSION_BUILD" in line:
                build = line.split("=")[1].strip()
            elif "VERSION_ALPHA" in line:
                alpha = line.split("=")[1].strip()
            if (major and minor and build and alpha) or "# END_VERSION_BLOCK" in line:
                break
    version = f"{major}.{minor}.{build}"
    if alpha and int(alpha) > 0:
        version += f"a{alpha}"
    return version


def get_requirements(requirements_filename: str):
    requirements_file = path.join(BASEDIR, requirements_filename)
    with open(requirements_file, "r", encoding="utf-8") as r:
        requirements = r.readlines()
    return [r.strip() for r in requirements if r.strip() and not r.strip().startswith("#")]


def find_resource_files():
    resource_base_dirs = ("locale",)
    package_data = ["*.json"]
    for res in resource_base_dirs:
        if path.isdir(path.join(BASEDIR, res)):
            for directory, _, files in walk(path.join(BASEDIR, res)):
                if files:
                    package_data.append(path.join(directory.replace(BASEDIR, "").lstrip("/"), "*"))
    return package_data


with open(path.join(BASEDIR, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name=PYPI_NAME,
    version=get_version(),
    description="OVOS pipeline plugin that reinforces good manners and reprimands foul language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Source": URL,
        "Bug Tracker": f"{URL}/issues",
    },
    author="OpenVoiceOS",
    author_email="jarbasai@mailfence.com",
    license="Apache-2.0",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    package_dir={PLUGIN_PKG: "."},
    package_data={PLUGIN_PKG: find_resource_files()},
    packages=[PLUGIN_PKG],
    include_package_data=True,
    install_requires=get_requirements("requirements.txt"),
    keywords="ovos pipeline plugin voice assistant manners",
    entry_points={"opm.pipeline": PLUGIN_ENTRY_POINT},
)
