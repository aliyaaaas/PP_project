package ru.itis.pp_project.lsh;

import ru.itis.pp_project.benchmark.MemoryEstimator;
import ru.itis.pp_project.distance.DistanceMetric;
import ru.itis.pp_project.index.ANNIndex;
import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;

import java.util.*;

public class LSHSequential implements ANNIndex {
    protected List<Vector> dataset = List.of();
    protected RandomHyperplane[] hashers;
    protected List<Map<Long, List<Integer>>> tables;
    protected final int bits, tableCount;
    protected final long seed;

    public LSHSequential(int bits, int tableCount, long seed) {
        if (bits < 1 || bits > 64 || tableCount < 1) throw new IllegalArgumentException("Invalid LSH parameters");
        this.bits = bits; this.tableCount = tableCount; this.seed = seed;
    }

    @Override public String name() { return "LSH-Sequential"; }

    @Override public void build(List<Vector> data) {
        dataset = List.copyOf(data);
        int dim = dataset.get(0).values().length;
        hashers = new RandomHyperplane[tableCount];
        tables = new ArrayList<>(tableCount);
        for (int t = 0; t < tableCount; t++) {
            hashers[t] = new RandomHyperplane(bits, dim, seed + t * 104729L);
            Map<Long, List<Integer>> table = new HashMap<>();
            for (int i = 0; i < dataset.size(); i++) {
                long hash = hasherHash(t, dataset.get(i).values());
                table.computeIfAbsent(hash, ignored -> new ArrayList<>()).add(i);
            }
            tables.add(table);
        }
    }

    private long hasherHash(int t, float[] v) {
        return hashers[t].hash(v);
    }

    @Override public List<SearchResult> search(Vector query, int k) {
        if (tables == null) throw new IllegalStateException("Index is not built");
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        BitSet seen = new BitSet(dataset.size());
        PriorityQueue<SearchResult> best = new PriorityQueue<>(k, Comparator.reverseOrder());
        for (int t = 0; t < tableCount; t++) {
            List<Integer> candidates = tables.get(t).get(hashers[t].hash(query.values()));
            if (candidates == null) continue;
            for (int index : candidates) if (!seen.get(index)) {
                seen.set(index);
                Vector v = dataset.get(index);
                SearchResult r = new SearchResult(v.id(), DistanceMetric.squaredL2(query.values(), v.values()));
                if (best.size() < k) best.add(r);
                else if (r.compareTo(best.peek()) < 0) { best.poll(); best.add(r); }
            }
        }
        List<SearchResult> result = new ArrayList<>(best);
        result.sort(Comparator.naturalOrder());
        return result;
    }

    @Override public List<List<SearchResult>> searchBatch(List<Vector> queries, int k) {
        List<List<SearchResult>> out = new ArrayList<>(queries.size());
        for (Vector q : queries) out.add(search(q, k));
        return out;
    }

    @Override public long estimateIndexMemoryBytes() {
        if (dataset.isEmpty() || tables == null) return 0L;
        return MemoryEstimator.estimateLSHBytes(dataset.size(), dataset.get(0).values().length, bits, tables);
    }
}