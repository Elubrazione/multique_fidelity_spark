import json

with open("huge_space.json.origin", "r") as f:
    data = json.load(f)

    default_config = {'spark.broadcast.blockSize': 7, 'spark.default.parallelism': 495, 'spark.driver.cores': 1, 'spark.driver.maxResultSize': 3484, 'spark.driver.memory': 75, 'spark.driver.memoryOverhead': 8266, 'spark.executor.cores': 14, 'spark.executor.heartbeatInterval': 91, 'spark.executor.instances': 9, 'spark.executor.memory': 93, 'spark.executor.memoryOverhead': 16133, 'spark.files.fetchTimeout': 119, 'spark.files.maxPartitionBytes': 330115300, 'spark.files.openCostInBytes': 9186861, 'spark.io.compression.snappy.blockSize': 92, 'spark.kryoserializer.buffer': 45, 'spark.kryoserializer.buffer.max': 949, 'spark.locality.wait': 5, 'spark.memory.fraction': 0.30000000000000004, 'spark.memory.offHeap.size': 5, 'spark.memory.storageFraction': 0.9, 'spark.network.timeout': 14839, 'spark.network.timeoutInterval': 217, 'spark.reducer.maxSizeInFlight': 191, 'spark.scheduler.excludeOnFailure.unschedulableTaskSetTimeout': 220, 'spark.scheduler.maxRegisteredResourcesWaitingTime': 18, 'spark.scheduler.minRegisteredResourcesRatio': 0.55, 'spark.scheduler.revive.interval': 2, 'spark.shuffle.accurateBlockThreshold': 48673750, 'spark.shuffle.file.buffer': 42, 'spark.shuffle.mapOutput.minSizeForBroadcast': 1959, 'spark.shuffle.registration.maxAttempts': 1, 'spark.shuffle.registration.timeout': 4111, 'spark.shuffle.service.index.cache.size': 270, 'spark.shuffle.spill.diskWriteBufferSize': 50250071, 'spark.shuffle.unsafe.file.output.buffer': 201, 'spark.speculation.interval': 255, 'spark.speculation.quantile': 0.25, 'spark.sql.adaptive.rebalancePartitionsSmallPartitionFactor': 0.1, 'spark.sql.autoBroadcastJoinThreshold': 515, 'spark.sql.broadcastTimeout': 8574, 'spark.sql.files.maxPartitionBytes': 101434143, 'spark.sql.shuffle.partitions': 357, 'spark.sql.sources.parallelPartitionDiscovery.parallelism': 69, 'spark.stage.maxConsecutiveAttempts': 5, 'spark.storage.localDiskByExecutors.cacheSize': 698, 'spark.storage.memoryMapThreshold': 4, 'spark.storage.unrollMemoryThreshold': 4743675, 'spark.task.cpus': 6, 'spark.task.maxFailures': 1, 'spark.task.reaper.pollingInterval': 49}

    for key, value in default_config.items():
        data[key]["default"] = value

    with open("huge_space.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Default values updated successfully.")