import os
from setuptools import setup, find_packages
from yaml import load
import jinja2


def load_meta_data(fname):
    with open(fname) as input_fp:
        # note that EDITABLE_FLAG is not required because we getting it from environ
        template = jinja2.Template(input_fp.read(), undefined=jinja2.StrictUndefined)
        return load(template.render(**os.environ))


def main():
    meta_data = load_meta_data('conda-build/meta.yaml')
    setup(
        name=meta_data['package']['name'],
        version=meta_data['package']['version'],
        include_package_data=True,
        zip_safe=False,
        entry_points={
            'console_scripts': meta_data['build'].get('entry_points', [])
        },
        packages=find_packages()
    )


if __name__ == '__main__':
    main()
