from setuptools import find_packages, setup

package_name = 'ftc_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml','setup.cfg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ayush Kumar',
    maintainer_email='211230013@nitdelhi.ac.in',
    description='A Fault-Tolerant Control Node for UAVs in PX4-Gazebo-ROS2 simulation.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ftc_node = ftc_node.ftc_node:main'
        ],
    },
)
