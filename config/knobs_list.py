from enum import Enum
from copy import deepcopy
from config.config import data_size, workload

MEMORY_MIN = 0
CORE_MIN = 0
MEMORY_MAX = 192000
CORE_MAX = 144


class KnobType(Enum):
    INTEGER = 'integer'
    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'


class Unit(Enum):
    GB = 'g'
    MB = 'm'
    KB = 'k'
    MILLISECOND = 'ms'
    SECOND = 's'


RESOURCE_KNOB_DETAILS = {
    # Number of cores by driver process
    'spark.driver.cores': {
        'type': KnobType.INTEGER,
        'range': [1, 16, 1],
        'default': 1,
        'range_adjustable': False,
        'limit_exceed': [False, False],
    },
    # Memory size for driver process
    'spark.driver.memory': {
        'type': KnobType.INTEGER,
        'range': [10, 120, 1],
        'default': 20,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.GB.value
    },
    # Number of cores per executor
    'spark.executor.cores': {
        'type': KnobType.INTEGER,
        'range': [1, 32, 1],
        'default': 10,
        'range_adjustable': False,
        'limit_exceed': [False, False],
    },
    # Total number of Executor processes used for the Spark job
    'spark.executor.instances': {
        'type': KnobType.INTEGER,
        'range': [1, 24, 1],
        'default': 2,
        'range_adjustable': False,
        'limit_exceed': [False, False],
    },
    # Memory size per executor process
    'spark.executor.memory': {
        'type': KnobType.INTEGER,
        'range': [1, 180, 1],
        'default': 10,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.GB.value
    },
    # Memory size which can be used for off-heap allocation
    'spark.memory.offHeap.size': {
        'type': KnobType.INTEGER,
        'range': [0, 10, 1],
        'default': 1,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.GB.value
    }
}
# if data_size < 50:
#     RESOURCE_KNOB_DETAILS['spark.driver.cores']['range'] = [2, 8, 1]
#     RESOURCE_KNOB_DETAILS['spark.driver.memory']['range'] = [4 * 1024, 12 * 1024, 64]
#     RESOURCE_KNOB_DETAILS['spark.executor.cores']['range'] = [2, 8, 1]
#     RESOURCE_KNOB_DETAILS['spark.executor.instances']['range'] = [2, 8, 1]
#     RESOURCE_KNOB_DETAILS['spark.executor.memory']['range'] = [4 * 1024, 12 * 1024, 64]
# elif data_size > 250:
#     RESOURCE_KNOB_DETAILS['spark.driver.cores']['range'] = [3, 12, 1]
#     RESOURCE_KNOB_DETAILS['spark.driver.memory']['range'] = [4 * 1024, 20 * 1024, 64]
#     RESOURCE_KNOB_DETAILS['spark.executor.cores']['range'] = [3, 12, 1]
#     RESOURCE_KNOB_DETAILS['spark.executor.instances']['range'] = [3, 12, 1]
#     RESOURCE_KNOB_DETAILS['spark.executor.memory']['range'] = [4 * 1024, 20 * 1024, 64]
#     RESOURCE_KNOB_DETAILS['spark.memory.offHeap.size']['range'] = [1 * 1024, 4 * 1024, 64]


NON_RESOURCE_KNOB_DETAILS = {
    # Size of each piece of a block for TorrentBroadcastFactory
    'spark.broadcast.blockSize': {
        'type': KnobType.INTEGER,
        'range': [1, 32, 1],
        'default': 4,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    # Number of RDD partitions
    'spark.default.parallelism': {
        'type': KnobType.INTEGER,
        'range': [24, 3000, 1],
        'default': 100,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    # Memory overhead of each executor
    'spark.executor.memoryOverhead': {
        'type': KnobType.INTEGER,
        'range': [384, 20 * 1024, 64],
        'default': 384,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    # Wait time to launch task in data-local before in a less-local node
    'spark.locality.wait': {
        'type': KnobType.INTEGER,
        'range': [0, 10, 1],
        'default': 3,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    # Fraction for execution and storage memory
    'spark.memory.fraction': {
        'type': KnobType.NUMERIC,
        'range': [0.1, 0.9, 0.01],
        'default': 0.6,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    # Storage memory percent exempt from eviction
    'spark.memory.storageFraction': {
        'type': KnobType.NUMERIC,
        'range': [0.1, 0.9, 0.01],
        'default': 0.5,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    # Max map outputs to collect concurrently per reduce task
    'spark.reducer.maxSizeInFlight': {
        'type': KnobType.INTEGER,
        'range': [1, 300, 1],
        'default': 48,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    # In-memory buffer size per output stream
    'spark.shuffle.file.buffer': {
        'type': KnobType.INTEGER,
        'range': [1, 300, 1],
        'default': 32,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.KB.value
    },
    # Specifies the maximum size for a broadcasted table
    'spark.sql.autoBroadcastJoinThreshold': {
        'type': KnobType.INTEGER,
        'range': [8, 3000 if workload != 'IMDB' else 16, 1],
        'default': 10,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    # Default partition number when shuffling data for join/aggregations
    'spark.sql.shuffle.partitions': {
        'type': KnobType.INTEGER,
        'range': [24, 3000, 1],
        'default': 200,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    # Specifies mapped memory size when read a block from the disk
    'spark.storage.memoryMapThreshold': {
        'type': KnobType.INTEGER,
        'range': [1, 10, 1],
        'default': 2,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    # 以下是新增的配置参数
    'spark.task.cpus': {
        'type': KnobType.INTEGER,
        'range': [1, 8, 1],
        'default': 6,
        'range_adjustable': False,
        'limit_exceed': [False, False],
    },
    'spark.network.timeout': {
        'type': KnobType.INTEGER,
        'range': [120, 30000, 1],
        'default': 14839,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.sql.broadcastTimeout': {
        'type': KnobType.INTEGER,
        'range': [300, 30000, 1],
        'default': 8574,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.sql.sources.parallelPartitionDiscovery.parallelism': {
        'type': KnobType.INTEGER,
        'range': [10, 500, 1],
        'default': 69,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.driver.maxResultSize': {
        'type': KnobType.INTEGER,
        'range': [2048, 6144, 1],
        'default': 3484,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    'spark.driver.memoryOverhead': {
        'type': KnobType.INTEGER,
        'range': [384, 20480, 1],
        'default': 8266,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    'spark.shuffle.unsafe.file.output.buffer': {
        'type': KnobType.INTEGER,
        'range': [1, 300, 1],
        'default': 201,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.KB.value
    },
    'spark.shuffle.spill.diskWriteBufferSize': {
        'type': KnobType.INTEGER,
        'range': [1048576, 104857600, 1],
        'default': 50250071,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.shuffle.service.index.cache.size': {
        'type': KnobType.INTEGER,
        'range': [1, 300, 1],
        'default': 270,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    'spark.shuffle.accurateBlockThreshold': {
        'type': KnobType.INTEGER,
        'range': [1048576, 314572800, 1],
        'default': 48673750,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.shuffle.registration.timeout': {
        'type': KnobType.INTEGER,
        'range': [1000, 10000, 1],
        'default': 4111,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.shuffle.registration.maxAttempts': {
        'type': KnobType.INTEGER,
        'range': [1, 5, 1],
        'default': 1,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.shuffle.mapOutput.minSizeForBroadcast': {
        'type': KnobType.INTEGER,
        'range': [100, 3000, 1],
        'default': 1959,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    'spark.io.compression.snappy.blockSize': {
        'type': KnobType.INTEGER,
        'range': [1, 96, 1],
        'default': 92,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.KB.value
    },
    'spark.kryoserializer.buffer.max': {
        'type': KnobType.INTEGER,
        'range': [1, 1024, 1],
        'default': 949,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    'spark.kryoserializer.buffer': {
        'type': KnobType.INTEGER,
        'range': [1, 300, 1],
        'default': 45,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.KB.value
    },
    'spark.storage.unrollMemoryThreshold': {
        'type': KnobType.INTEGER,
        'range': [1048576, 8388608, 1],
        'default': 4743675,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.storage.localDiskByExecutors.cacheSize': {
        'type': KnobType.INTEGER,
        'range': [100, 3000, 1],
        'default': 698,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MB.value
    },
    'spark.executor.heartbeatInterval': {
        'type': KnobType.INTEGER,
        'range': [5, 100, 1],
        'default': 91,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.files.fetchTimeout': {
        'type': KnobType.INTEGER,
        'range': [1, 300, 1],
        'default': 119,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.SECOND.value
    },
    'spark.files.maxPartitionBytes': {
        'type': KnobType.INTEGER,
        'range': [10485760, 524288000, 1],
        'default': 330115300,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.files.openCostInBytes': {
        'type': KnobType.INTEGER,
        'range': [1048576, 10485760, 1],
        'default': 9186861,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.network.timeoutInterval': {
        'type': KnobType.INTEGER,
        'range': [30, 600, 1],
        'default': 217,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.scheduler.maxRegisteredResourcesWaitingTime': {
        'type': KnobType.INTEGER,
        'range': [10, 120, 1],
        'default': 18,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.SECOND.value
    },
    'spark.scheduler.revive.interval': {
        'type': KnobType.INTEGER,
        'range': [1, 10, 1],
        'default': 2,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.SECOND.value
    },
    'spark.scheduler.excludeOnFailure.unschedulableTaskSetTimeout': {
        'type': KnobType.INTEGER,
        'range': [100, 600, 1],
        'default': 220,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.SECOND.value
    },
    'spark.speculation.interval': {
        'type': KnobType.INTEGER,
        'range': [100, 1000, 1],
        'default': 255,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.task.maxFailures': {
        'type': KnobType.INTEGER,
        'range': [1, 10, 1],
        'default': 1,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.task.reaper.pollingInterval': {
        'type': KnobType.INTEGER,
        'range': [5, 60, 1],
        'default': 49,
        'range_adjustable': False,
        'limit_exceed': [False, False],
        'unit': Unit.MILLISECOND.value
    },
    'spark.stage.maxConsecutiveAttempts': {
        'type': KnobType.INTEGER,
        'range': [1, 10, 1],
        'default': 5,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.sql.files.maxPartitionBytes': {
        'type': KnobType.INTEGER,
        'range': [10485760, 524288000, 1],
        'default': 101434143,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.speculation.quantile': {
        'type': KnobType.NUMERIC,
        'range': [0.1, 1.0, 0.01],
        'default': 0.25,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor': {
        'type': KnobType.NUMERIC,
        'range': [0.1, 1.0, 0.01],
        'default': 0.1,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    },
    'spark.scheduler.minRegisteredResourcesRatio': {
        'type': KnobType.NUMERIC,
        'range': [0.5, 1.0, 0.01],
        'default': 0.55,
        'range_adjustable': False,
        'limit_exceed': [False, False]
    }
}

KNOB_DETAILS = deepcopy(NON_RESOURCE_KNOB_DETAILS)
KNOB_DETAILS.update(RESOURCE_KNOB_DETAILS)
KNOBS = sorted(list(KNOB_DETAILS.keys()))

EXTRA_KNOBS = {
    'spark.master': 'yarn',
    'spark.submit.deployMode': 'cluster',
    'spark.eventLog.enabled': 'true',
    'spark.eventLog.compress': 'false',
    'spark.yarn.jars': 'jar-path',
    'spark.eventLog.dir': 'event-log-path',
    'spark.yarn.maxAppAttempts': 1,
    'spark.sql.catalogImplementation': 'hive',
    'spark.memory.offHeap.enabled': 'true',
    'spark.sql.sources.parallelPartitionDiscovery.parallelism': 270
}