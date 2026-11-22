package ru.itis.pp_project.lsh;

import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public final class LSHParallel extends LSHSequential {
    private final ForkJoinPool pool;
    public LSHParallel(int bits, int tableCount, long seed, int threads) {
        super(bits, tableCount, seed);
        if (threads < 1) throw new IllegalArgumentException("threads must be positive");
        pool = new ForkJoinPool(threads);
    }
    @Override public String name() { return "LSH-Parallel-" + pool.getParallelism(); }
    @Override public void build(List<Vector> data) {
        dataset = List.copyOf(data);
        int dim = dataset.get(0).values().length;
        hashers = new RandomHyperplane[tableCount];
        @SuppressWarnings("unchecked") Map<Long, List<Integer>>[] built = new Map[tableCount];
        pool.invoke(new RecursiveAction() {
            @Override protected void compute() {
                List<RecursiveAction> tasks = new ArrayList<>();
                for (int t = 0; t < tableCount; t++) {
                    int tableId = t;
                    tasks.add(new RecursiveAction() {
                        @Override protected void compute() {
                            RandomHyperplane hasher = new RandomHyperplane(bits, dim, seed + tableId * 104729L);
                            hashers[tableId] = hasher;
                            Map<Long, List<Integer>> table = new HashMap<>();
                            for (int i = 0; i < dataset.size(); i++) {
                                long hash = hasher.hash(dataset.get(i).values());
                                table.computeIfAbsent(hash, ignored -> new ArrayList<>()).add(i);
                            }
                            built[tableId] = table;
                        }
                    });
                }
                invokeAll(tasks);
            }
        });
        tables = List.of(built);
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
