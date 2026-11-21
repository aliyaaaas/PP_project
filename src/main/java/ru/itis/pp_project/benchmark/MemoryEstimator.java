package ru.itis.pp_project.benchmark;

import ru.itis.pp_project.model.Vector;

import java.util.List;
import java.util.Map;

/** Approximate memory estimates for raw vector payload and ANN index structures. */
public final class MemoryEstimator {
    private MemoryEstimator() {}

    public static long vectorPayloadBytes(List<Vector> vectors) {
        long bytes = 0;
        for (Vector v : vectors) bytes += 4L * v.values().length;
        return bytes;
    }

    public static long estimateIVFBytes(int n, int dim, int nlist) {
        long centroidsBytes = (long) nlist * (16L + (long) dim * Float.BYTES);
        // ArrayList<List<Integer>> + boxed Integer references (16B Integer object + 4B pointer)
        long listsBytes = 24L + (long) nlist * 24L + (long) n * 20L;
        return centroidsBytes + listsBytes;
    }

    public static long estimateLSHBytes(int n, int dim, int bits, List<Map<Long, List<Integer>>> tables) {
        if (tables == null) return 0L;
        int l = tables.size();
        long hyperplanesBytes = (long) l * bits * (16L + (long) dim * Float.BYTES);
        long uniqueBuckets = 0;
        for (Map<Long, List<Integer>> table : tables) uniqueBuckets += table.size();
        // HashMap.Node (32B) + Long key (24B) + ArrayList header (24B)
        long bucketHeadersBytes = uniqueBuckets * 80L;
        // Each vector ID is duplicated into all L tables as boxed Integer (20B)
        long duplicatedIdsBytes = (long) l * n * 20L;
        return hyperplanesBytes + bucketHeadersBytes + duplicatedIdsBytes;
    }

    public static long heapUsedBytes() {
        Runtime r = Runtime.getRuntime();
        return r.totalMemory() - r.freeMemory();
    }
}