import logging
import os
import random
from collections.abc import Callable
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from anemoi.training.data.grid_indices import BaseGridIndices
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData
import nvidia.dali as dali
import nvidia.dali.plugin.pytorch as dali_pth

LOGGER = logging.getLogger(__name__)

@dataclass
class MetaData(DatapipeMetaData):
    name: str = "ERA5ZarrDALI"
    auto_device: bool = True
    cuda_graphs: bool = True
    ddp_sharding: bool = True

class NativeGridDatapipe(Datapipe):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
        batch_size: int = 1,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        prefetch_factor: int = 2,
        val_rollout: int = 1,
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        batch_size : int, optional
            batch size, by default 1
        num_workers : int, optional
            number of workers, by default 1
        device : str or torch.device, optional
            device to use, by default "cuda"
        prefetch_factor : int, optional
            prefetch factor, by default 2
        val_rollout : int, optional
            number of validation rollouts, by default 1
        """
        
        super().__init__(meta=MetaData())
        
        self.label = label
        self.data = data_reader

        self.timestep = timestep
        self.grid_indices = grid_indices

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        # normalize device
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.val_rollout = val_rollout
        # Data dimensions
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

        # relative index of dates to extract
        self.relative_date_indices = relative_date_indices

        # expose dataset for distributed strategy compatibility
        self.dataset = self

        # lazy pipeline handle
        self.pipe = None

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i.
        """
        return get_usable_indices(
            self.data.missing,
            len(self.data),
            np.array(self.relative_date_indices, dtype=np.int64),
            self.data.trajectory_ids,
        )

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size
        shard_end = (self.sample_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.info(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, seed group id %d, base_seed %d, sanity rnd %f)"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            base_seed,
            sanity_rnd,
        )
    
    def _create_pipeline(self) -> dali.Pipeline:
        device_id = (self.device.index if self.device.type == "cuda" else 0) or 0
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=4,
            prefetch_queue_depth=self.prefetch_factor,
            py_num_workers=self.num_workers,
            device_id=device_id,
            py_start_method="spawn",
            exec_async=True,
            exec_pipelined=True,
        )

        with pipe:
            source = NativeGridExternalSource(
                data=self.data,
                shuffle=self.shuffle,
                rng=self.rng,
                valid_date_indices=self.valid_date_indices,
                chunk_index_range=self.chunk_index_range,
                label=self.label,
                worker_id=self.worker_id,
                global_rank=self.global_rank,
                model_comm_group_id=self.model_comm_group_id,
                model_comm_group_rank=self.model_comm_group_rank,
                sample_comm_group_id=self.sample_comm_group_id,
                relative_date_indices=self.relative_date_indices,
                grid_indices=self.grid_indices,
                reader_group_rank=self.reader_group_rank,
                ensemble_dim=self.ensemble_dim,
                num_batches=len(self.chunk_index_range) // self.batch_size if self.chunk_index_range is not None else len(self.valid_date_indices) // self.batch_size,
            )

            x = dali.fn.external_source(source, parallel=True, batch=False)
            if self.device.type == "cuda":
                x = x.gpu()
            pipe.set_outputs(x)

        return pipe

    def __iter__(self):
        if self.chunk_index_range is None:
            # single-worker initialization when used as a standalone dataloader
            self.per_worker_init(n_workers=1, worker_id=0)
        if self.pipe is None:
            self.pipe = self._create_pipeline()
        self.pipe.reset()
        dali_iter = dali_pth.DALIGenericIterator([self.pipe], output_map=["x"], auto_reset=True)
        for batch in dali_iter:
            yield batch[0]["x"]

    def __len__(self):
        if self.chunk_index_range is None:
            self.per_worker_init(n_workers=1, worker_id=0)
        return int(len(self.chunk_index_range) // self.batch_size)


class NativeGridExternalSource:
    def __init__(
        self,
        data,
        shuffle,
        rng,
        valid_date_indices,
        chunk_index_range,
        label,
        worker_id,
        global_rank,
        model_comm_group_id,
        model_comm_group_rank,
        sample_comm_group_id,
        relative_date_indices,
        grid_indices,
        reader_group_rank,
        ensemble_dim,
        num_batches,
    ):
        self.data = data
        self.shuffle = shuffle
        self.rng = rng
        self.valid_date_indices = valid_date_indices
        self.chunk_index_range = chunk_index_range
        self.label = label
        self.worker_id = worker_id
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.sample_comm_group_id = sample_comm_group_id
        self.relative_date_indices = relative_date_indices
        self.grid_indices = grid_indices
        self.reader_group_rank = reader_group_rank
        self.ensemble_dim = ensemble_dim
        self.num_batches = num_batches

        if self.shuffle:
            self.shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            self.shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[np.ndarray]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()
        i = self.shuffled_chunk_indices[sample_info.idx_in_epoch]
        start = i + self.relative_date_indices[0]
        end = i + self.relative_date_indices[-1] + 1
        timeincrement = self.relative_date_indices[1] - self.relative_date_indices[0]
        # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
        # data[start...] will be replaced with data[self.relative_date_indices + i]

        grid_shard_indices = self.grid_indices.get_shard_indices(self.reader_group_rank)
        if isinstance(grid_shard_indices, slice):
            # Load only shards into CPU memory
            x = self.data[start:end:timeincrement, :, :, grid_shard_indices]

        else:
            # Load full grid in CPU memory, select grid_shard after
            # Note that anemoi-datasets currently doesn't support slicing + indexing
            # in the same operation.
            x = self.data[start:end:timeincrement, :, :, :]
            x = x[..., grid_shard_indices]  # select the grid shard
        # collapse ensemble dimension by taking ensemble index 0
        # output shape: [dates, gridpoints, variables]
        x = rearrange(x, "dates variables ensemble gridpoints -> dates gridpoints variables")
        return x.astype(np.float32)
