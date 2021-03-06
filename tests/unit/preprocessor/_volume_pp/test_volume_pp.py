"""Unit test for the :func:`esmvaltool.preprocessor._volume_pp` function"""

from __future__ import absolute_import, division, print_function

import unittest

import iris
import numpy as np
from cf_units import Unit

import tests
from esmvaltool.preprocessor._volume_pp import volume_slice
from esmvaltool.preprocessor._volume_pp import volume_average
from esmvaltool.preprocessor._volume_pp import depth_integration
from esmvaltool.preprocessor._volume_pp import extract_transect
from esmvaltool.preprocessor._volume_pp import extract_trajectory


class Test(tests.Test):
    """Test class for _volume_pp"""

    def setUp(self):
        """Prepare tests"""
        coord_sys = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
        data1 = np.ones((3, 2, 2))
        data2 = np.ma.ones((2, 3, 2, 2))
        data3 = np.ma.ones((4, 3, 2, 2))
        mask3 = np.full((4, 3, 2, 2), False)
        mask3[0, 0, 0, 0] = True
        data3 = np.ma.array(data3, mask=mask3)

        time = iris.coords.DimCoord(
            [15, 45],
            standard_name='time',
            bounds=[[1., 30.], [30., 60.]],
            units=Unit('days since 1950-01-01', calendar='gregorian'))
        time2 = iris.coords.DimCoord(
            [1., 2., 3., 4.],
            standard_name='time',
            bounds=[[0.5, 1.5], [1.5, 2.5], [2.5, 3.5], [3.5, 4.5], ],
            units=Unit('days since 1950-01-01', calendar='gregorian'))

        depth = iris.coords.DimCoord(
            [0.5, 5., 50.],
            standard_name='depth',
            bounds=[[0., 2.5], [2.5, 25.], [25., 250.]],
            units='m')
        lons2 = iris.coords.DimCoord(
            [1.5, 2.5],
            standard_name='longitude',
            bounds=[[1., 2.], [2., 3.]],
            units='degrees_east',
            coord_system=coord_sys)
        lats2 = iris.coords.DimCoord(
            [1.5, 2.5],
            standard_name='latitude',
            bounds=[[1., 2.], [2., 3.]],
            units='degrees_north',
            coord_system=coord_sys)

        coords_spec3 = [(depth, 0), (lats2, 1), (lons2, 2)]
        self.grid_3d = iris.cube.Cube(data1, dim_coords_and_dims=coords_spec3)

        coords_spec4 = [(time, 0), (depth, 1), (lats2, 2), (lons2, 3)]
        self.grid_4d = iris.cube.Cube(data2, dim_coords_and_dims=coords_spec4)

        coords_spec5 = [(time2, 0), (depth, 1), (lats2, 2), (lons2, 3)]
        self.grid_4d_2 = iris.cube.Cube(data3,
                                        dim_coords_and_dims=coords_spec5)

    def test_volume_slice(self):
        """Test to extract the top two layers of a 3 layer depth column."""
        result = volume_slice(self.grid_3d, 0., 10.)
        expected = np.ones((2, 2, 2))
        print(result.data, expected.data)
        self.assertArrayEqual(result.data, expected)

    def test_volume_average(self):
        """Test to take the volume weighted average of a (2,3,2,2) cube."""
        result = volume_average(self.grid_4d, 'depth', 'latitude', 'longitude')
        expected = np.array([1., 1.])
        self.assertArrayEqual(result.data, expected)

    def test_volume_average_long(self):
        """
        Test to take the volume weighted average of a (4,3,2,2) cube.

        This extra time is needed, as the volume average calculation uses
        different methods for small and large cubes.
        """
        result = volume_average(self.grid_4d_2,
                                'depth',
                                'latitude',
                                'longitude')
        expected = np.array([1., 1., 1., 1.])
        self.assertArrayEqual(result.data, expected)

    def test_depth_integration_1d(self):
        """Test to take the depth integration of a 3 layer cube."""
        result = depth_integration(self.grid_3d[:, 0, 0], 'depth')
        expected = np.ones((1, 1)) * 250.
        print(result.data, expected.data)
        self.assertArrayEqual(result.data, expected)

    def test_depth_integration_3d(self):
        """Test to take the depth integration of a 3 layer cube."""
        result = depth_integration(self.grid_3d, 'depth')
        expected = np.ones((2, 2)) * 250.
        print(result.data, expected.data)
        self.assertArrayEqual(result.data, expected)

    def test_extract_transect_latitude(self):
        """Test to extract a transect from a (3, 2, 2) cube."""
        result = extract_transect(self.grid_3d, latitude=1.5)
        expected = np.ones((3, 2))
        self.assertArrayEqual(result.data, expected)

    def test_extract_transect_longitude(self):
        """Test to extract a transect from a (3, 2, 2) cube."""
        result = extract_transect(self.grid_3d, longitude=1.5)
        expected = np.ones((3, 2))
        self.assertArrayEqual(result.data, expected)

    def test_extract_trajectory(self):
        """Test to extract a trajectory from a (3, 2, 2) cube."""
        result = extract_trajectory(self.grid_3d, [1.5, 2.5], [2., 2.], 2)
        expected = np.ones((3, 2))
        self.assertArrayEqual(result.data, expected)


if __name__ == '__main__':
    unittest.main()
