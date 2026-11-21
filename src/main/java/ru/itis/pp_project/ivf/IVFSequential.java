package ru.itis.pp_project.ivf;

import ru.itis.pp_project.benchmark.MemoryEstimator;
import ru.itis.pp_project.distance.DistanceMetric;
import ru.itis.pp_project.index.ANNIndex;
import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class IVFSequential implements ANNIndex {
    protected List<Vector> dataset = List.of();
    protected float[][] centroids;
    protected List<List<Integer>> lists;
    protected final int nlist, nprobe, iterations;
    protected final long seed;

    public IVFSequential(int nlist, int nprobe, int iterations, long seed) {
        if (nlist < 1 || nprobe < 1 || nprobe > nlist) throw new IllegalArgumentException("Require 1 <= nprobe <= nlist");
        this.nlist = nlist; this.nprobe = nprobe; this.iterations = iterations; this.seed = seed;
    }

    @Override public String name() { return "IVF-Sequential"; }

    @Override public void build(List<Vector> data) {
        dataset = List.copyOf(data);
        centroids = KMeans.fit(dataset, Math.min(nlist, dataset.size()), iterations, seed);
        buildLists();
    }

    protected void buildLists() {
        lists = new ArrayList<>(centroids.length);
        for (int i = 0; i < centroids.length; i++) lists.add(new ArrayList<>());
        for (int i = 0; i < dataset.size(); i++) lists.get(nearestCentroid(dataset.get(i).values())).add(i);
    }

    protected int nearestCentroid(float[] v) {
        int best = 0; float bestD = Float.POSITIVE_INFINITY;
        for (int c = 0; c < centroids.length; c++) {
            float d = DistanceMetric.squaredL2(v, centroids[c]);
            if (d < bestD) { bestD = d; best = c; }
        }
        return best;
    }

    @Override public List<SearchResult> search(Vector query, int k) {
        if (lists == null) throw new IllegalStateException("Index is not built");
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        PriorityQueue<SearchResult> best = new PriorityQueue<>(k, Comparator.reverseOrder());
        int[] selected = nearestCentroids(query.values());
        for (int c : selected) for (int index : lists.get(c)) {
            Vector v = dataset.get(index);
            SearchResult r = new SearchResult(v.id(), DistanceMetric.squaredL2(query.values(), v.values()));
            if (best.size() < k) best.add(r);
            else if (r.compareTo(best.peek()) < 0) { best.poll(); best.add(r); }
        }
        List<SearchResult> result = new ArrayList<>(best);
        result.sort(Comparator.naturalOrder());
        return result;
    }

    protected int[] nearestCentroids(float[] query) {
        int count = Math.min(nprobe, centroids.length);
        SearchResult[] distances = new SearchResult[centroids.length];
        for (int c = 0; c < centroids.length; c++) distances[c] = new SearchResult(c, DistanceMetric.squaredL2(query, centroids[c]));
        java.util.Arrays.sort(distances);
        int[] result = new int[count];
        for (int i = 0; i < count; i++) result[i] = distances[i].vectorId();
        return result;
    }

    @Override public List<List<SearchResult>> searchBatch(List<Vector> queries, int k) {
        List<List<SearchResult>> out = new ArrayList<>(queries.size());
        for (Vector q : queries) out.add(search(q, k));
        return out;
    }

    @Override public long estimateIndexMemoryBytes() {
        if (dataset.isEmpty() || centroids == null) return 0L;
        return MemoryEstimator.estimateIVFBytes(dataset.size(), dataset.get(0).values().length, centroids.length);
    }
}