# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections.abc import Callable
from functools import cached_property

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData

from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDatapipe
from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)

class DummyDataset:
    def __init__(self, data):
        self.data = data
        self.name_to_index = data.name_to_index

class DaliDataModule(pl.LightningDataModule):
    """Anemoi Datasets data module for PyTorch Lightning."""

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        """Initialize Anemoi Datasets data module.

        Parameters
        ----------
        config : BaseSchema
            Job configuration

        """
        super().__init__()

        self.config = config
        self.graph_data = graph_data

        # Set the training end date if not specified
        if self.config.dataloader.training.end is None:
            LOGGER.info(
                "No end date specified for training data, setting default before validation start date %s.",
                self.config.dataloader.validation.start - 1,
            )
            self.config.dataloader.training.end = self.config.dataloader.validation.start - 1

        if not self.config.dataloader.pin_memory:
            LOGGER.info("Data loader memory pinning disabled.")

        self.data = open_dataset(self.config.dataloader.training)

    @cached_property
    def statistics(self) -> dict:
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        try:
            return self.data.statistics_tendencies(self.config.data.timestep)
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays() | self.grid_indices.supporting_arrays

    @cached_property
    def data_indices(self) -> IndexCollection:
        return IndexCollection(self.config, self.data.name_to_index)

    def relative_date_indices(self, val_rollout: int = 1) -> list:
        """Determine a list of relative time indices to load for each batch."""
        if hasattr(self.config.training, "explicit_times"):
            return sorted(set(self.config.training.explicit_times.input + self.config.training.explicit_times.target))

        # Calculate indices using multistep, timeincrement and rollout.
        # Use the maximum rollout to be expected
        rollout_cfg = getattr(getattr(self.config, "training", None), "rollout", None)

        rollout_max = getattr(rollout_cfg, "max", None)
        rollout_start = getattr(rollout_cfg, "start", 1)
        rollout_epoch_increment = getattr(rollout_cfg, "epoch_increment", 0)

        # Fallback if max is None or rollout_cfg is missing
        rollout_value = rollout_start
        if rollout_cfg and rollout_epoch_increment > 0 and rollout_max is not None:
            rollout_value = rollout_max

        else:
            LOGGER.warning(
                "Falling back rollout to: %s",
                rollout_value,
            )

        rollout = max(rollout_value, val_rollout)

        multi_step = self.config.training.multistep_input
        return [self.timeincrement * mstep for mstep in range(multi_step + rollout)]

    def add_trajectory_ids(self, data_reader: Callable) -> Callable:
        """Determine an index of forecast trajectories associated with the time index and add to a data_reader object.

        This is needed for interpolation to ensure that the interpolator is trained on consistent time slices.

        NOTE: This is only relevant when training on non-analysis and could in the future be replaced with
        a property of the dataset stored in data_reader. Now assumes regular interval of changed model runs
        """
        if not hasattr(self.config.dataloader, "model_run_info"):
            data_reader.trajectory_ids = None
            return data_reader

        mr_start = np.datetime64(self.config.dataloader.model_run_info.start)
        mr_len = self.config.dataloader.model_run_info.length  # model run length in number of date indices
        if hasattr(self.config.training, "rollout") and self.config.training.rollout.max is not None:
            max_rollout_index = max(self.relative_date_indices(self.config.training.rollout.max))
            assert (
                max_rollout_index < mr_len
            ), f"""Requested data length {max_rollout_index + 1}
                    longer than model run length {mr_len}"""

        data_reader.trajectory_ids = (data_reader.dates - mr_start) // np.timedelta64(
            mr_len * frequency_to_seconds(self.config.data.frequency),
            "s",
        )
        return data_reader

    @cached_property
    def grid_indices(self) -> type[BaseGridIndices]:
        reader_group_size = self.config.dataloader.read_group_size

        grid_indices = instantiate(
            self.config.dataloader.grid_indices,
            reader_group_size=reader_group_size,
        )
        grid_indices.setup(self.graph_data)
        return grid_indices

    @cached_property
    def timeincrement(self) -> int:
        """Determine the step size relative to the data frequency."""
        try:
            frequency = frequency_to_seconds(self.config.data.frequency)
        except ValueError as e:
            msg = f"Error in data frequency, {self.config.data.frequency}"
            raise ValueError(msg) from e

        try:
            timestep = frequency_to_seconds(self.config.data.timestep)
        except ValueError as e:
            msg = f"Error in timestep, {self.config.data.timestep}"
            raise ValueError(msg) from e

        assert timestep % frequency == 0, (
            f"Timestep ({self.config.data.timestep} == {timestep}) isn't a "
            f"multiple of data frequency ({self.config.data.frequency} == {frequency})."
        )

        LOGGER.info(
            "Timeincrement set to %s for data with frequency, %s, and timestep, %s",
            timestep // frequency,
            frequency,
            timestep,
        )
        return timestep // frequency

    @cached_property
    def ds_train(self):
        return DummyDataset(open_dataset(self.config.dataloader.training))
    
    @cached_property
    def ds_valid(self):
        return DummyDataset(open_dataset(self.config.dataloader.validation))
    
    @cached_property
    def ds_test(self):
        return DummyDataset(open_dataset(self.config.dataloader.test))

    def _get_dataloader(self, stage: str) -> DataLoader:
        assert stage in {"training", "validation", "test"}
        if stage == "training":
            data = open_dataset(self.config.dataloader.training)
            label = "train"
            val_rollout = 1
            shuffle = True
        elif stage == "validation":
            data = open_dataset(self.config.dataloader.validation)
            label = "validation"
            shuffle = False
            val_rollout = self.config.dataloader.validation_rollout
        elif stage == "test":
            data = open_dataset(self.config.dataloader.test)
            label = "test"
            shuffle = False
            val_rollout = 1

        data = self.add_trajectory_ids(data)
        self.relative_date_indices(val_rollout)
        dp = NativeGridDatapipe(
            data_reader=data,
            grid_indices=self.grid_indices,
            relative_date_indices=self.relative_date_indices(val_rollout),
            timestep=self.config.data.timestep,
            shuffle=shuffle,
            label=label,
            batch_size=self.config.dataloader.batch_size[stage],
            num_workers=self.config.dataloader.num_workers[stage],
            prefetch_factor=self.config.dataloader.prefetch_factor,
        )
        # expose dataset for distributed strategy compatibility
        if not hasattr(dp, "dataset"):
            dp.dataset = dp
        return dp

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")
