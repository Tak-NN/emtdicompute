"""Split closest-hit queries over a committed, read-only Embree scene."""
from concurrent.futures import ThreadPoolExecutor
import operator

import numpy as np

MIN_RAYS_PER_TASK = 65536


def validate_num_workers(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('num_workers must be a positive integer.')
    try:
        workers = operator.index(value)
    except TypeError:
        raise ValueError('num_workers must be a positive integer.') from None
    if workers < 1:
        raise ValueError('num_workers must be a positive integer.')
    return workers


class EmbreeQueries:
    """Reuse one thread pool across viewpoints; preserve original ray order."""
    def __init__(self, num_workers):
        self.num_workers = validate_num_workers(num_workers)
        self.pool = None

    def __enter__(self):
        if self.num_workers > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix='emtdi-embree')
        return self.run

    def __exit__(self, *args):
        if self.pool is not None:
            self.pool.shutdown(wait=True)

    def run(self, scene, origins, directions):
        count = len(origins)
        tasks = min(self.num_workers, max(1, (count + MIN_RAYS_PER_TASK - 1) // MIN_RAYS_PER_TASK))
        if tasks == 1:
            return scene.scene.run(origins, directions, query='INTERSECT', output=True)
        # Embree's lazy commit must finish before querying the shared scene from
        # worker threads. Every worker then has independent input/output arrays.
        if not scene.scene.is_committed:
            scene.scene.run(origins[:1], directions[:1], query='INTERSECT', output=True)
        boundaries = np.linspace(0, count, tasks + 1, dtype=np.int64)
        spans = list(zip(boundaries[:-1], boundaries[1:]))

        def block(span):
            start, end = span
            return scene.scene.run(origins[start:end], directions[start:end], query='INTERSECT', output=True)

        ids = np.empty(count, dtype=np.int32)
        distances = np.empty(count, dtype=np.float32)
        for (start, end), result in zip(spans, self.pool.map(block, spans)):
            ids[start:end] = result['primID']
            distances[start:end] = result['tfar']
        return {'primID': ids, 'tfar': distances}
