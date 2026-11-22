package ru.itis.pp_project.ivf;

import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public final class IVFParallel extends IVFSequential {
    private final ForkJoinPool pool;
    public IVFParallel(int nlist, int nprobe, int iterations, long seed, int threads) {
        super(nlist, nprobe, iterations, seed);
        if (threads < 1) throw new IllegalArgumentException("threads must be positive");
        pool = new ForkJoinPool(threads);
    }
    @Override public String name() { return "IVF-Parallel-" + pool.getParallelism(); }
    @Override public void build(List<Vector> data) {
        dataset = List.copyOf(data);
        centroids = KMeans.fitParallel(dataset, Math.min(nlist, dataset.size()), iterations, seed, pool);
        buildLists();
    }
    @Override public List<List<SearchResult>> searchBatch(List<Vector> queries, int k) {
        @SuppressWarnings("unchecked") List<SearchResult>[] results = new List[queries.size()];
        pool.invoke(new RecursiveAction() {
            @Override protected void compute() {
                int p = pool.getParallelism();
                int chunk = Math.max(1, (queries.size() + p * 4 - 1) / (p * 4));
                List<RecursiveAction> tasks = new ArrayList<>();
                for (int from = 0; from < queries.size(); from += chunk) {
                    int start = from, end = Math.min(queries.size(), from + chunk);
                    tasks.add(new RecursiveAction() {
                        @Override protected void compute() {
                            for (int i = start; i < end; i++) results[i] = search(queries.get(i), k);
                        }
                    });
                }
                invokeAll(tasks);
            }
        });
        return List.of(results);
    }
    @Override public void close() { pool.shutdown(); }
}
