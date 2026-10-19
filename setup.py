import os
import os.path as osp
import platform
import shutil
import sys
import warnings
from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
version_ns = {}
exec((root / 'mmpose' / 'version.py').read_text(encoding='utf-8'), version_ns)


def add_mim_extension():
    """Expose MMPose resources under ``mmpose/.mim`` when packaging.

    This follows the upstream MMPose packaging convention so MIM/MMEngine
    inferencer utilities can find configs, demos and metafiles.
    """
    if 'develop' in sys.argv:
        mode = 'copy' if platform.system() == 'Windows' else 'symlink'
    elif 'sdist' in sys.argv or 'bdist_wheel' in sys.argv:
        mode = 'copy'
    else:
        return

    filenames = [
        'tools', 'configs', 'demo', 'model-index.yml', 'dataset-index.yml'
    ]
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, 'mmpose', '.mim')
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if not osp.exists(filename):
            continue
        src_path = osp.join(repo_path, filename)
        tar_path = osp.join(mim_path, filename)
        if osp.isfile(tar_path) or osp.islink(tar_path):
            os.remove(tar_path)
        elif osp.isdir(tar_path):
            shutil.rmtree(tar_path)

        if mode == 'symlink':
            src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
            os.symlink(src_relpath, tar_path)
        elif mode == 'copy':
            if osp.isfile(src_path):
                shutil.copyfile(src_path, tar_path)
            elif osp.isdir(src_path):
                shutil.copytree(src_path, tar_path)
            else:
                warnings.warn(f'Cannot copy file {src_path}.')


if __name__ == '__main__':
    add_mim_extension()
    setup(
        name='mmpose-hpp',
        version=version_ns['__version__'],
        description='Hierarchical Pose Prompting for Generalized Animal Pose Estimation',
        long_description=(root / 'README.md').read_text(encoding='utf-8'),
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        include_package_data=True,
        python_requires='>=3.8',
        zip_safe=False,
    )
