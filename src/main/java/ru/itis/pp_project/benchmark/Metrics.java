package ru.itis.pp_project.benchmark;

import ru.itis.pp_project.model.SearchResult;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

public final class Metrics {
    private Metrics() {}
    public static double recallAtK(List<SearchResult> exact, List<SearchResult> approximate, int k) {
        if (exact.isEmpty() || k <= 0) return 0.0;
        Set<Integer> ids = new HashSet<>();
        for (int i = 0; i < Math.min(k, exact.size()); i++) ids.add(exact.get(i).vectorId());
        int found = 0;
        for (int i = 0; i < Math.min(k, approximate.size()); i++) if (ids.contains(approximate.get(i).vectorId())) found++;
        return (double) found / Math.min(k, exact.size());
    }
    public static double speedup(long sequentialNanos, long parallelNanos) {
        return parallelNanos <= 0 ? 0.0 : (double) sequentialNanos / parallelNanos;
    }
    public static double efficiency(double speedup, int threads) { return threads <= 0 ? 0.0 : speedup / threads; }
    public static double qps(int queries, long nanos) { return nanos <= 0 ? 0.0 : queries * 1_000_000_000.0 / nanos; }
}
