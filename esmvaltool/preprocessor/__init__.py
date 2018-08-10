"""Preprocessor module."""
import logging
import os

from iris.cube import Cube
from prov.dot import prov_to_dot
from prov.model import ProvDocument

from .._task import BaseTask
from ._area_pp import area_average as average_region
from ._area_pp import area_slice as extract_region
from ._area_pp import zonal_means
from ._derive import derive
from ._download import download
from ._io import cleanup, concatenate, extract_metadata, load_cubes, save
from ._mask import (mask_above_threshold, mask_below_threshold,
                    mask_fillvalues, mask_inside_range, mask_landsea,
                    mask_outside_range)
from ._multimodel import multi_model_statistics
from ._reformat import (cmor_check_data, cmor_check_metadata, fix_data,
                        fix_file, fix_metadata)
from ._regrid import regrid
from ._regrid import vinterp as extract_levels
from ._time_area import (extract_month, extract_season, seasonal_mean,
                         time_average)
from ._time_area import time_slice as extract_time
from ._volume_pp import depth_integration, extract_trajectory, extract_transect
from ._volume_pp import volume_average as average_volume
from ._volume_pp import volume_slice as extract_volume

logger = logging.getLogger(__name__)

__all__ = [
    'download',
    # File reformatting/CMORization
    'fix_file',
    # Load cube from file
    'load_cubes',
    # Derive variable
    'derive',
    # Metadata reformatting/CMORization
    'fix_metadata',
    # Concatenate all cubes in one
    'concatenate',
    'cmor_check_metadata',
    # Time extraction
    'extract_time',
    'extract_season',
    'extract_month',
    # Data reformatting/CMORization
    'fix_data',
    # Level extraction
    'extract_levels',
    # Mask landsea (fx or Natural Earth)
    'mask_landsea',
    # Regridding
    'regrid',
    # Masking missing values
    'mask_fillvalues',
    'mask_above_threshold',
    'mask_below_threshold',
    'mask_inside_range',
    'mask_outside_range',
    # Region selection
    'extract_region',
    'extract_volume',
    'extract_trajectory',
    'extract_transect',
    # 'average_zone': average_zone,
    # 'cross_section': cross_section,
    # Time operations
    # 'annual_cycle': annual_cycle,
    # 'diurnal_cycle': diurnal_cycle,
    'multi_model_statistics',
    # Grid-point operations
    'depth_integration',
    'average_region',
    'average_volume',
    'zonal_means',
    'seasonal_mean',
    'time_average',
    'cmor_check_data',
    # Save to file
    'save',
    'cleanup',
    'extract_metadata',
]

DEFAULT_ORDER = tuple(__all__)
assert set(DEFAULT_ORDER).issubset(set(globals()))

INITIAL_STEPS = DEFAULT_ORDER[:DEFAULT_ORDER.index('fix_data') + 1]
FINAL_STEPS = DEFAULT_ORDER[DEFAULT_ORDER.index('cmor_check_data'):]

MULTI_MODEL_FUNCTIONS = {
    'multi_model_statistics',
    'mask_fillvalues',
    'extract_metadata',
}
assert MULTI_MODEL_FUNCTIONS.issubset(set(DEFAULT_ORDER))

# Preprocessor functions that take a list instead of a file/Cube as input.
_LIST_INPUT_FUNCTIONS = MULTI_MODEL_FUNCTIONS | {
    'download',
    'load_cubes',
    'concatenate',
    'derive',
    'save',
    'cleanup',
}
assert _LIST_INPUT_FUNCTIONS.issubset(set(DEFAULT_ORDER))

# Preprocessor functions that return a list instead of a file/Cube.
_LIST_OUTPUT_FUNCTIONS = MULTI_MODEL_FUNCTIONS | {
    'download',
    'load_cubes',
    'save',
    'cleanup',
}
assert _LIST_OUTPUT_FUNCTIONS.issubset(set(DEFAULT_ORDER))


def split_settings(settings, step, order=DEFAULT_ORDER):
    """Split settings, using step as a separator."""
    before = {}
    for _step in order:
        if _step == step:
            break
        if _step in settings:
            before[_step] = settings[_step]
    after = {
        k: v
        for k, v in settings.items() if not (k == step or k in before)
    }
    return before, after


def _get_multi_model_settings(all_settings, step):
    """Select settings for multi model step"""
    for settings in all_settings.values():
        if step in settings:
            return {step: settings[step]}
    return None


def _group_input(in_files, out_files):
    """Group a list of input files by output file."""
    grouped_files = {}

    def get_matching(in_file):
        """Find the output file which matches input file best."""
        in_chunks = os.path.basename(in_file).split('_')
        score = 0
        fname = []
        for out_file in out_files:
            out_chunks = os.path.basename(out_file).split('_')
            tmp = sum(c in out_chunks for c in in_chunks)
            if tmp > score:
                score = tmp
                fname = [out_file]
            elif tmp == score:
                fname.append(out_file)
        if not fname:
            logger.warning(
                "Unable to find matching output file for input file %s",
                in_file)
        return fname

    # Group input files by output file
    for in_file in in_files:
        for out_file in get_matching(in_file):
            if out_file not in grouped_files:
                grouped_files[out_file] = []
            grouped_files[out_file].append(in_file)

    return grouped_files


def preprocess_multi_model(input_files, all_settings, order, debug=False):
    """Run preprocessor on multiple models for a single variable."""
    # Group input files by output file
    all_items = _group_input(input_files, all_settings)
    doc = ProvDocument()
    doc.add_namespace('evt', 'http://www.esmvaltool.org/scheme')
    for name in all_items:
        all_items[name] = ProductList.from_files(all_items[name], doc)
    logger.debug("Processing %s", all_items)

    # List of all preprocessor steps used
    steps = [
        step for step in order
        if any(step in settings for settings in all_settings.values())
    ]
    # Find multi model steps
    # This assumes that the multi model settings are the same for all models
    multi_model_steps = [
        step for step in steps if step in MULTI_MODEL_FUNCTIONS
    ]
    # Append a dummy multi model step if the final step is not multi model
    dummy_step = object()
    if steps[-1] not in MULTI_MODEL_FUNCTIONS:
        multi_model_steps.append(dummy_step)

    # Process
    for step in multi_model_steps:
        multi_model_settings = _get_multi_model_settings(all_settings, step)
        # Run single model steps
        for name in all_settings:
            settings, all_settings[name] = split_settings(
                all_settings[name], step, order)
            all_items[name] = preprocess(all_items[name], settings, order,
                                         debug)
        if step is not dummy_step:
            # Run multi model step
            multi_model_items = ProductList(
                [item for name in all_items for item in all_items[name]], doc)
            all_items = {}
            result = preprocess(multi_model_items, multi_model_settings, order,
                                debug)
            for product in result:
                if isinstance(product.data, Cube):
                    name = product.data.attributes['_filename']
                    if name not in all_items:
                        all_items[name] = ProductList([], doc)
                    all_items[name].append(product)
                else:
                    all_items[product.data] = ProductList([product], doc)
    filenames = [
        product.data for name in all_items for product in all_items[name]
    ]
    return filenames, doc


def preprocess(products, settings, order, debug=False):
    """Run preprocessor"""
    steps = (step for step in order if step in settings)
    for step in steps:
        logger.debug("Running preprocessor step %s", step)
        products = products.apply(step, settings[step])

        if debug:
            items = [p.data for p in products]
            logger.debug("Result %s", items)
            cubes = [item for item in items if isinstance(item, Cube)]
            save(cubes, debug=debug, step=step)

    return products


def write_provenance(provenance, output_dir):
    """Write provenance information to output_dir."""
    filename = os.path.join(output_dir, 'provenance')
    logger.info("Writing provenance to %s.xml", filename)
    provenance.serialize(filename + '.xml', format='xml')

    graph = prov_to_dot(provenance)
    logger.info("Writing provenance to %s.png", filename)
    graph.write_png(filename + '.png')
    logger.info("Writing provenance to %s.pdf", filename)
    graph.write_pdf(filename + '.pdf')


class Product(object):
    def __init__(self, data, entity, doc):
        self.data = data
        self.entity = entity
        self._doc = doc

    @classmethod
    def from_file(cls, filename, doc):
        entity = doc.entity('evt:' + filename)
        return cls(filename, entity, doc)

    def apply(self, step, args):

        # Do the computation
        function = globals()[step]
        logger.debug("Running %s(%s, %s)", function.__name__, self.data, args)
        result = function(self.data, **args)
        if step not in _LIST_OUTPUT_FUNCTIONS:
            result = [result]

        # Track provenance
        args = {'evt:' + k: str(v) for k, v in args.items()}
        activity = self._doc.activity(
            'evt:' + step + ':' + str(self), other_attributes=args)
        products = []
        for item in result:
            entity = self._doc.entity('evt:' + str(self) + ';' + step)
            entity.wasDerivedFrom(self.entity, activity)
            products.append(Product(item, entity, self._doc))

        return products

    def __str__(self):
        return str(self.entity.identifier)[4:]


class ProductList(object):
    def __init__(self, products, doc):
        self.products = products
        self._doc = doc
        self.entity = doc.collection('evt:' + '+'.join(
            str(p) for p in products))
        for product in self.products:
            self.entity.hadMember(product.entity)

    @classmethod
    def from_files(cls, filenames, doc):
        products = [Product.from_file(f, doc) for f in filenames]
        return cls(products, doc)

    def apply(self, step, args):

        products = []
        if step not in _LIST_INPUT_FUNCTIONS:
            for product in self.products:
                products.extend(product.apply(step, args))
        else:
            # Do the computation
            function = globals()[step]
            items = [p.data for p in self.products]
            logger.debug("Running %s(%s, %s)", function.__name__, items, args)
            result = function(items, **args)
            if step not in _LIST_OUTPUT_FUNCTIONS:
                result = [result]

            # Track provenance
            args = {'evt:' + k: str(v) for k, v in args.items()}
            name = str(self.entity.identifier)[4:]
            activity = self._doc.activity(
                'evt:' + step + ':' + name, other_attributes=args)
            for i, item in enumerate(result):
                entity = self._doc.entity('evt:' + name + ';' + step + '-' +
                                          str(i))
                entity.wasDerivedFrom(self.entity, activity)
                products.append(Product(item, entity, self._doc))

        return ProductList(products, self._doc)

    def append(self, value):
        self.products.append(value)

    def __iter__(self):
        return iter(self.products)

    def __str__(self):
        return '[' + '\n'.join(str(p) for p in self.products) + ']'


class PreprocessingTask(BaseTask):
    """Task for running the preprocessor"""

    def __init__(self,
                 settings,
                 output_dir,
                 ancestors=None,
                 input_files=None,
                 order=DEFAULT_ORDER,
                 debug=None):
        """Initialize"""
        super(PreprocessingTask, self).__init__(
            settings=settings, output_dir=output_dir, ancestors=ancestors)
        self.order = list(order)
        self.debug = debug
        self._input_files = input_files

    def _run(self, input_files):
        # If input_data is not available from ancestors and also not
        # specified in self.run(input_files), use default
        if not self.ancestors and not input_files:
            input_files = self._input_files
        output_files, provenance = preprocess_multi_model(
            input_files, self.settings, self.order, debug=self.debug)
        write_provenance(provenance, self.output_dir)
        return output_files

    def __str__(self):
        """Get human readable description."""
        settings = dict(self.settings)
        self.settings = {
            os.path.basename(k): v
            for k, v in self.settings.items()
        }

        txt = "{}:\norder: {}\n{}".format(
            self.__class__.__name__,
            tuple(
                step for step in self.order
                if any(step in settings for settings in settings.values())),
            super(PreprocessingTask, self).str(),
        )

        self.settings = settings

        if self._input_files is not None:
            txt += '\ninput_files: {}'.format(self._input_files)
        return txt
