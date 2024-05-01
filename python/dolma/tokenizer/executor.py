import sys
import importlib
import hashlib
import multiprocessing
import os
import random
import tempfile
from contextlib import ExitStack
from math import ceil, log10
from queue import Queue  # pylint: disable=unused-import
from typing import Any, Dict, Generator, List, Optional
#
import time
import tqdm
import pickle
import smart_open
from threading import Thread
from datetime import datetime

import numpy as np
from typing_extensions import TypeAlias

from transformers import AutoTokenizer

from ..core.loggers import get_logger
from ..core.parallel import BaseParallelProcessor, QueueType
from ..core.paths import get_size, glob_path, join_path, mkdir_p, parent
from .data_types import TokenizerOutput  # pylint: disable=unused-import
from .memmap_writer import MemmapWriter
from .tokenizer import Tokenizer, tokenize_file

MPI = None
comm = None
rank = 0
worldsize = 1

logging_flag = np.zeros(10, dtype=int)

TokenizedSeqsQueueType: TypeAlias = "Queue[List[TokenizerOutput]]"
PathsQueueType: TypeAlias = "Queue[str]"


def import_tokenizer_class(path):
  if os.path.isdir(path):
    path = os.path.join(path, "tokenizer.py")
  if not os.path.exists(path):
    raise OSError(f"File {path} not exists.")
  module_name = "tokenizer"
  spec = importlib.util.spec_from_file_location(module_name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module.Tokenizer


def sizes_to_probs(sizes: List[int]) -> np.ndarray:
    return np.array(sizes) / sum(sizes)


class MemMapParallelWriter(BaseParallelProcessor):

    @classmethod
    def increment_progressbar(  # type: ignore[override]    # pylint: disable=arguments-differ
        cls,
        queue: QueueType,
        /,
        files: int = 0,
        documents: int = 0,
        tokens: int = 0,
        memmaps: int = 0,
    ) -> Dict[str, int]:
        return super().increment_progressbar(
            queue, files=files, documents=documents, tokens=tokens, memmaps=memmaps
        )

    @classmethod
    def process_single(cls, source_path: str, destination_path: str, queue: QueueType, **kwargs: Any):
        logger = get_logger(__name__)

        max_size: int = kwargs.pop("max_size", None) or 1024 * 1024 * 1024
        dtype: np.dtype = np.dtype(kwargs.pop("dtype", None) or "uint16")
        local_shuffle: int = kwargs.pop("local_shuffle", None) or 10_000
        ring_size: int = kwargs.pop("ring_size", None) or 8
        sample_ring_prop: bool = kwargs.pop("sample_ring_prop", None) or False

        global_source_paths = kwargs.pop("grouped_source_prefixes", None)
        if not isinstance(global_source_paths, list):
            raise RuntimeError("grouped_source_prefixes should be a list of paths")
        elif len(global_source_paths) == 0:
            raise RuntimeError("grouped_source_prefixes should not be empty")
        source_paths = global_source_paths[int(source_path)]

        tokenizer_name_or_path = kwargs.pop("tokenizer_name_or_path", None)
        if tokenizer_name_or_path is None:
            raise RuntimeError("tokenizer_name_or_path not provided")

        tokenizer_kwargs = {}
        tokenizer_kwargs["bos_token_id"] = kwargs.pop("bos_token_id", None)
        tokenizer_kwargs["eos_token_id"] = kwargs.pop("eos_token_id", None)
        if tokenizer_kwargs["bos_token_id"] is None and tokenizer_kwargs["eos_token_id"] is None:
            raise ValueError(
                "Neither eos_token_id nor bos_token_id specified. "
                "At least one of them should be provided; otherwise, documents will not be properly separated."
            )

        tokenizer_kwargs["pad_token_id"] = kwargs.pop("pad_token_id", None)
        if tokenizer_kwargs["pad_token_id"] is None:
            logger.warning("pad_token_id not provided, using eos_token_id")
            tokenizer_kwargs["pad_token_id"] = tokenizer_kwargs["eos_token_id"]

        # flag to control whether to segment the documents before tokenization
        tokenizer_kwargs["segment_before_tokenization"] = kwargs.pop("segment_before_tokenization", None) or False

        # this is useful for making sure the queue does not grows too much
        cpu_count = multiprocessing.cpu_count()

        # these are used to keep track of the progress
        documents_cnt = tokens_cnt = 0
        update_interval = 1
        mm_cnt = 0

        if logging_flag[0] == 0:
            print(f"Using dtype={dtype} for tokens.")
            logging_flag[0] += 1

        # create the tokenizer from file if it exists, otherwise from pretrained
        if kwargs.get("tiktoken", ""):
            if rank == 0 and logging_flag[1] == 0:
                print("Using tiktoken tokenizer")
                logging_flag[1] += 1
            tokenizer = import_tokenizer_class(kwargs["tiktoken"])(tokenizer_name_or_path)
        elif kwargs.pop("auto_tokenizer", False):
            if rank == 0 and logging_flag[2] == 0:
                print("Using HF auto tokenizer")
                logging_flag[2] += 1
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        else:
            if os.path.exists(tokenizer_name_or_path) and os.path.isfile(tokenizer_name_or_path):
                tokenizer = Tokenizer.from_file(tokenizer_name_or_path, **tokenizer_kwargs)
            else:
                tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)

        tokenizer_ring: List[Generator[TokenizerOutput, None, None]] = []
        tokenizer_sizes: List[int] = []
        for _ in range(min(ring_size, len(source_paths))):
            path = source_paths.pop()
            tokenizer_ring.append(tokenize_file(tokenizer=tokenizer, path=path))
            tokenizer_sizes.append(get_size(path))

        # this is the probabilities with which we sample from the ring buffer if sample_ring_prop is True
        tokenizer_probs = sizes_to_probs(tokenizer_sizes)

        accumulator = []

        with ExitStack() as stack:
            memwriter = stack.enter_context(
                MemmapWriter(path=destination_path + f"-{mm_cnt:05d}", dtype=dtype, max_tokens=max_size)
            )
            cls.increment_progressbar(queue, memmaps=1)

            while len(source_paths) > 0 or len(tokenizer_ring) > 0:
                for i in range(local_shuffle):
                    if sample_ring_prop:
                        # you are sampling proportionally to the size of files in the ring
                        j = np.random.choice(len(tokenizer_ring), p=tokenizer_probs)
                    else:
                        # you are going round robin
                        j = i % len(tokenizer_ring)

                    try:
                        # trying to read the next sequence of tokens (might fail if end of file)
                        content = next(tokenizer_ring[j])

                        # added to the accumulator, we will shuffle this later
                        accumulator.append(content)

                        # count the number of tokens and documents
                        tokens_cnt += content.end - content.start
                        documents_cnt += 1
                    except StopIteration:
                        # we have reached the end of one of the file; move to the next!
                        cls.increment_progressbar(queue, files=1)
                        tokenizer_ring.pop(j)
                        tokenizer_sizes.pop(j)

                        if len(tokenizer_ring) == 0:
                            # break if no more files to tokenize
                            break
                        if len(source_paths) > 0:
                            path = source_paths.pop()
                            tokenizer_ring.append(tokenize_file(tokenizer=tokenizer, path=path))
                            tokenizer_sizes.append(get_size(path))

                        # wether a file is added or not to the ring, we must re-balance probabilities
                        tokenizer_probs = sizes_to_probs(tokenizer_sizes)

                    # check if it time to update the progress bar!
                    if documents_cnt >= update_interval:
                        cls.increment_progressbar(queue, documents=documents_cnt, tokens=tokens_cnt)
                        tokens_cnt = documents_cnt = 0

                        if queue.qsize() >= cpu_count:
                            # double the update interval if the queue is full
                            update_interval *= 2

                # shuffle sequence order to ensure that the sequences are well mixed
                random.shuffle(accumulator)

                # try to write all the sequences, collect the ones that don't fit in remaining
                remaining = memwriter.write_many(outputs=accumulator, flush=documents_cnt == 0)

                if remaining:
                    # if we have remaining sequences, we need to close the current memwriter and open a new one
                    mm_cnt += 1
                    stack.pop_all().close()
                    memwriter = stack.enter_context(
                        MemmapWriter(
                            path=destination_path + f"-{mm_cnt:05d}",
                            dtype=dtype,  # pyright: ignore
                            max_tokens=max_size,
                        )
                    )
                    cls.increment_progressbar(queue, memmaps=1)

                    # finally, write the remaining sequences
                    memwriter.write_many(outputs=remaining, flush=True)

                accumulator = []

                memwriter.flush()

        cls.increment_progressbar(queue, documents=documents_cnt, tokens=tokens_cnt)

    def __call__(self, num_readers: Optional[int] = None, use_mpi=False, **process_single_kwargs: Any):
        """Run the processor."""

        # get all source paths; shuffle them well
        all_source_paths = [p for source in self.src_prefixes for p in glob_path(source)]
        random.shuffle(all_source_paths)

        # TRICKY BIT: Group source paths into buckets
        # First, check what the step size should be. The step is the minimum between the
        # number of readers requested, and the number of source_paths per process.
        # The float("inf") bit is required to handle the case when num_readers is None.
        step_size = min(num_readers or float("inf"), len(all_source_paths) / self.num_processes)

        # Now, we step over all files in increments of step_size, and group them into buckets
        # we need to make sure we don't add empty buckets.
        grouped_source_prefixes: List[List[str]] = []
        current_step = 0.0
        while current_step < len(all_source_paths):  # can't use range here because of the float
            prefix_slice = all_source_paths[int(round(current_step)) : int(round(current_step + step_size))]
            if prefix_slice:
                grouped_source_prefixes.append(prefix_slice)
            current_step += step_size

        # Finally, we optionally redefine num_processes to be the number of groups otherwise some
        # processors will not have any work to do.
        self.num_processes = min(len(grouped_source_prefixes), self.num_processes)

        # We have one set of sanity checks here to make sure that the grouping was done correctly
        if any(len(bucket) == 0 for bucket in grouped_source_prefixes):
            raise ValueError("Some buckets are empty. This should not happen.")
        if len(grouped_source_prefixes) < self.num_processes:
            raise ValueError("The number of groups is less than the number of processes. This should not happen.")
        if len(all_source_paths) < len(grouped_source_prefixes):
            raise ValueError(
                "The number of groups is greater than the number of source paths. This should not happen."
            )
        if sum(len(bucket) for bucket in grouped_source_prefixes) != len(all_source_paths):
            raise ValueError(
                "The number of files in the groups does not match the total number of files. "
                "This should not happen."
            )

        # this is a bit of a hack but: we pass indices to grouped_source_prefixes to the processors
        # so that they can load the correct source paths
        source_indices = [str(i) for i in range(len(grouped_source_prefixes))]

        # check that only one value of destination and metadata is provided
        if len(set(self.dst_prefixes)) != 1 or len(set(self.meta_prefixes)) != 1:
            raise ValueError("Only one destination and metadata should be provided.")

        # make necessary destination directories
        destination = self.dst_prefixes[0]
        mkdir_p(destination)

        # each parallel processor will write a file name like part-dddddd-dddd.npy and part-dddddd-dddd.csv.gz
        digits = int(ceil(log10(len(grouped_source_prefixes) + 1)))
        all_destination_paths = [
            join_path(None, destination, f"part-{i:0{digits}d}") for i in range(len(grouped_source_prefixes))
        ]

        # same for metadata
        metadata = self.meta_prefixes[0]
        mkdir_p(metadata)
        all_metadata_path = [join_path(None, metadata, f"{i}.done") for i in range(len(all_destination_paths))]

        # give the user some feedback
        print(
            f"Tokenizing {sum(len(e) for e in grouped_source_prefixes):,} source files "
            f"into {len(grouped_source_prefixes):,} numpy destinations."
        )

        # finally run the processors
        if use_mpi:
            fn = self._mpi4py_run_all
        else:
            fn = self._debug_run_all if self.debug else self._multiprocessing_run_all
        fn(
            all_source_paths=source_indices,
            all_destination_paths=all_destination_paths,
            all_metadata_paths=all_metadata_path,
            grouped_source_prefixes=grouped_source_prefixes,
            **process_single_kwargs,
        )


    @classmethod
    def _mpi4py_process_single_and_save_status(
        cls,
        source_path: str,
        destination_path: str,
        metadata_path: str,
        #  kwargs,
        serialized_kwargs: bytes,
        queue: QueueType=None,
    ):
        """A wrapper around process single that saves a metadata file if processing is successful."""

        # make destination directory if it doesn't exist for the destination and metadata paths
        mkdir_p(parent(destination_path))
        mkdir_p(parent(metadata_path))

        kwargs = pickle.loads(serialized_kwargs)
        retries_on_error = kwargs.get("retries_on_error", 0) + 1
        while True:
            try:
                cls.process_single(
                    source_path=source_path, destination_path=destination_path, queue=queue, **kwargs
                )
                break
            except DolmaRetryableFailure as exception:
                retries_on_error -= 1
                if retries_on_error == 0:
                    raise DolmaError from exception

        # write the metadata file
        with smart_open.open(metadata_path, "wt") as f:
            f.write(datetime.now().isoformat())

        queue.put(None)
        counts = [0 for _ in cls.increment_progressbar(queue)]
        while True:
            item = queue.get()
            if item is None:
                break

            for i, value in enumerate(item):
              counts[i] += value
        return counts


    def _mpi4py_run_all(
        self,
        all_source_paths: List[str],
        all_destination_paths: List[str],
        all_metadata_paths: List[str],
        **process_single_kwargs: Any,
    ):
        """Run files in parallel using mpi4py.

        Args:
            all_source_paths (List[MultiPath]): The list of source paths to process.
            all_destination_paths (List[MultiPath]): The list of destination paths to save.
            all_metadata_paths (List[MultiPath]): The locations where to save metadata.
        """
        with ExitStack() as stack:
            pbar_queue = Queue()
            sample_queue_output = self.increment_progressbar(pbar_queue)
            #  pbars = [
            #      stack.enter_context(
            #          tqdm.tqdm(desc=str(k), unit=str(k)[:1], position=i, unit_scale=True)  # pyright: ignore
            #      )
            #      for i, k in enumerate(sample_queue_output)
            #  ]
            if rank == 0:
                thread = Thread(
                    target=self._run_mpi_threaded_progressbar, args=(pbar_queue, self.pbar_timeout), daemon=True
                )
                thread.start()
            #
            serialized_kwargs = pickle.dumps(process_single_kwargs)
            serialized_kwargs = comm.bcast(serialized_kwargs, root=0)
            all_source_paths = comm.bcast(all_source_paths, root=0)
            all_destination_paths = comm.bcast(all_destination_paths, root=0)
            all_metadata_paths = comm.bcast(all_metadata_paths, root=0)
            #
            self._mpi4py_process_single_and_save_status(
                queue=pbar_queue,
                source_path=all_source_paths[rank],
                destination_path=all_destination_paths[rank],
                metadata_path=all_metadata_paths[rank],
                #  kwargs=process_single_kwargs,
                serialized_kwargs=pickle.dumps(process_single_kwargs),
            )
            pbar_queue.put(None)
            if rank == 0:
                thread.join()


    @classmethod
    def _run_mpi_threaded_progressbar(
        cls,
        queue: QueueType,
        timeout: float,
    ):
        """Run a progress bar in a separate thread under MPI.

        Args:
            queue (QueueType): The queue to increment the progress bars.
            timeout (float): How often to update the progress bars in seconds.
        """

        sample_queue_output = cls.increment_progressbar(queue)


        with ExitStack() as stack:
            if rank == 0:
                pbars = [
                    stack.enter_context(
                        tqdm.tqdm(desc=str(k), unit=str(k)[:1], position=i, unit_scale=True)  # pyright: ignore
                    )
                    for i, k in enumerate(sample_queue_output)
                ]
            else:
                pbars = [0 for i, k in enumerate(sample_queue_output)]
            pvals = np.zeros(len(sample_queue_output), dtype="i")

            while True:
                item = queue.get()
                pvals[:] = 0

                if item is None:
                    break

                for i, value in enumerate(item):
                    pvals[i] += value

                #  comm.Allreduce(MPI.IN_PLACE, pvals, op=MPI.SUM)

                if rank == 0:
                    for pbar, value in zip(pbars, pvals):
                        pbar.update(value)

                time.sleep(timeout)
                #  comm.Barrier()


def tokenize_in_parallel(
    sources: List[str],
    destination: str,
    num_writers: int = 1,
    num_readers: Optional[int] = None,
    local_shuffle: int = 10_000,
    ring_size: int = 8,
    tokenizer_name_or_path: str = "allenai/gpt-neox-olmo-dolma-v1_5",
    bos_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = 50279,
    pad_token_id: Optional[int] = 1,
    segment_before_tokenization: bool = False,
    seed: int = 3920,
    metadata_dir: Optional[str] = None,
    max_size: int = 1024 * 1024 * 1024,
    dtype: str = "uint16",
    use_mpi: bool = False,
    debug: bool = False,
    sample_ring_prop: bool = False,
    auto_tokenizer: bool = False,
    tiktoken: str = "",
):
    """
    Tokenizes the input sources in parallel using multiple writers and readers.

    Args:
        sources (List[str]): List of source file paths to tokenize.
        destination (str): Destination directory to store the tokenized files.
        num_writers (int, optional): Number of writer processes to use. Defaults to 1.
        num_readers (int, optional): Number of reader processes to use. Defaults to None.
        local_shuffle (int, optional): Number of lines to shuffle locally before writing. Defaults to 10_000.
        ring_size (int, optional): Size of the ring buffer for inter-process communication. Defaults to 8.
        tokenizer_name_or_path (str, optional): Name or path of the tokenizer to use.
            Defaults to "allenai/gpt-neox-olmo-dolma-v1_5". Note that, if the tokenizer is changed,
            you may need to adjust `bos_token_id`, `eos_token_id`, and `pad_token_id` accordingly.
        bos_token_id (int, optional): ID of the beginning-of-sentence token. Defaults to None.
        eos_token_id (int, optional): ID of the end-of-sentence token. Defaults to 50279.
        pad_token_id (int, optional): ID of the padding token. Defaults to 1.
        segment_before_tokenization (bool, optional): Whether to segment the input before tokenization.
            Defaults to False.
        seed (int, optional): Seed value for randomization. Defaults to 3920.
        metadata_dir (str, optional): Directory to store metadata files. Defaults to None.
        max_size (int, optional): Maximum size of each tokenized file. Defaults to 1024 * 1024 * 1024.
        dtype (str, optional): Data type for tokenized files. Defaults to "uint16".
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        sample_ring_prop (bool, optional): Whether to sample from the ring buffer proportionally to the size
            of files. Otherwise, it will go round-robin. Defaults to False.
    """
    # variables to avoid issues with parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # do it once so it gets cached (unless it's local path, so no need)
    if not os.path.exists(tokenizer_name_or_path):
        Tokenizer.from_pretrained(
            identifier=tokenizer_name_or_path,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    # get a run hash
    run_hash = hashlib.sha256(("".join(sources) + tokenizer_name_or_path).encode("utf-8")).hexdigest()[:8]
    metadata_dir = metadata_dir or join_path(None, tempfile.gettempdir(), f"dolma-{run_hash}")


    if use_mpi:
        global MPI, comm, rank, worldsize
        from mpi4py import MPI as _MPI
        MPI = _MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        worldsize = comm.Get_size()
        if rank == 0:
            print("Using MPI for paralelization.")
        num_writers = worldsize


    parallel_writer = MemMapParallelWriter(
        source_prefix=sources,
        # the call action will actually get the first destination and
        # make relative paths from there. Unfortunately, BaseParallelProcessor
        # expects as many destinations as there are sources, so we employ
        # this "hack" (that is, repeating destination len(sources) times)
        # to get around that. Same thing applies to metadata_dir.
        destination_prefix=[destination for _ in sources],
        metadata_prefix=[metadata_dir for _ in sources],
        num_processes=num_writers,
        seed=seed,
        debug=debug,
    )
    parallel_writer(
        num_readers=num_readers,
        use_mpi=use_mpi,
        local_shuffle=local_shuffle,
        ring_size=ring_size,
        max_size=max_size,
        dtype=dtype,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        segment_before_tokenization=segment_before_tokenization,
        tokenizer_name_or_path=tokenizer_name_or_path,
        sample_ring_prop=sample_ring_prop,
        auto_tokenizer=auto_tokenizer,
        tiktoken=tiktoken,
    )
