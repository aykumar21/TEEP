from setuptools import setup

package_name = 'flood_detection_segmentation'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ayush Kumar',
    maintainer_email='211230013@nitdelhi.ac.in',
    description='Flood detection segmentation using a drone camera feed.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'collect_data = flood_detection_segmentation.collect_data:main',
        ],
    },
)
