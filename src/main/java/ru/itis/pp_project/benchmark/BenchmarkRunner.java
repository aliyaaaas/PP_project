package ru.itis.pp_project.benchmark;

import ru.itis.pp_project.data.Dataset;
import ru.itis.pp_project.exact.BruteForceIndex;
import ru.itis.pp_project.index.ANNIndex;
import ru.itis.pp_project.ivf.IVFParallel;
import ru.itis.pp_project.ivf.IVFSequential;
import ru.itis.pp_project.lsh.LSHParallel;
import ru.itis.pp_project.lsh.LSHSequential;
import ru.itis.pp_project.model.SearchResult;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class BenchmarkRunner {
    private static final int K = 10;
    private static final int WARMUP = 2;
    private static final int REPEATS = 5;

    public void run(Dataset dataset, int nlist, int nprobe, int bits, int tables, Path output) throws IOException {
        int[] threadCounts = {1, 2, 4, 8};
        BruteForceIndex exact = new BruteForceIndex();
        exact.build(dataset.vectors());
        List<List<SearchResult>> groundTruth = exact.searchBatch(dataset.queries(), K);

        CSVExporter.writeHeader(output,
                "algorithm,threads,n,dimension,queries,k,build_ms,search_ms,qps,recall_at_k," +
                        "build_speedup,build_efficiency,search_speedup,search_efficiency,vector_payload_bytes,index_memory_bytes");

        for (String algorithm : List.of("IVF", "LSH")) {
            long seqBuild = -1;
            long seqSearch = -1;
            for (int p : threadCounts) {
                ANNIndex index = algorithm.equals("IVF")
                        ? (p == 1 ? new IVFSequential(nlist, nprobe, 12, 42L) : new IVFParallel(nlist, nprobe, 12, 42L, p))
                        : (p == 1 ? new LSHSequential(bits, tables, 42L) : new LSHParallel(bits, tables, 42L, p));
                try (index) {
                    for (int w = 0; w < WARMUP; w++) {
                        index.build(dataset.vectors());
                        index.searchBatch(dataset.queries(), K);
                    }
                    long[] buildTimes = new long[REPEATS];
                    long[] searchTimes = new long[REPEATS];
                    List<List<SearchResult>> last = List.of();
                    for (int r = 0; r < REPEATS; r++) {
                        long start = System.nanoTime();
                        index.build(dataset.vectors());
                        buildTimes[r] = System.nanoTime() - start;

                        start = System.nanoTime();
                        last = index.searchBatch(dataset.queries(), K);
                        searchTimes[r] = System.nanoTime() - start;
                    }
                    long buildMedian = median(buildTimes);
                    long searchMedian = median(searchTimes);
                    double recall = averageRecall(groundTruth, last, K);

                    if (p == 1) {
                        seqBuild = buildMedian;
                        seqSearch = searchMedian;
                    }
                    double buildSpeedup = Metrics.speedup(seqBuild, buildMedian);
                    double buildEfficiency = Metrics.efficiency(buildSpeedup, p);
                    double searchSpeedup = Metrics.speedup(seqSearch, searchMedian);
                    double searchEfficiency = Metrics.efficiency(searchSpeedup, p);
                    long payloadBytes = MemoryEstimator.vectorPayloadBytes(dataset.vectors());
                    long indexBytes = index.estimateIndexMemoryBytes();

                    String row = String.format(java.util.Locale.ROOT,
                            "%s,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.5f,%.5f,%.5f,%.5f,%.5f,%d,%d",
                            algorithm, p, dataset.vectors().size(), dataset.vectors().get(0).values().length,
                            dataset.queries().size(), K, buildMedian / 1e6, searchMedian / 1e6,
                            Metrics.qps(dataset.queries().size(), searchMedian), recall,
                            buildSpeedup, buildEfficiency, searchSpeedup, searchEfficiency,
                            payloadBytes, indexBytes);
                    CSVExporter.append(output, row);

                    System.out.printf("%-4s P=%d build=%8.2f ms (S=%.2f) search=%8.2f ms (S=%.2f, E=%.2f) QPS=%8.1f Recall@%d=%.3f Mem=%.2f MB%n",
                            algorithm, p, buildMedian / 1e6, buildSpeedup,
                            searchMedian / 1e6, searchSpeedup, searchEfficiency,
                            Metrics.qps(dataset.queries().size(), searchMedian), K, recall,
                            indexBytes / (1024.0 * 1024.0));
                }
            }
        }
        System.out.println("CSV saved to: " + output.toAbsolutePath());
    }

    private static double averageRecall(List<List<SearchResult>> exact, List<List<SearchResult>> actual, int k) {
        double sum = 0;
        for (int i = 0; i < exact.size(); i++) sum += Metrics.recallAtK(exact.get(i), actual.get(i), k);
        return exact.isEmpty() ? 0 : sum / exact.size();
    }

    private static long median(long[] values) {
        List<Long> sorted = new ArrayList<>();
        for (long v : values) sorted.add(v);
        Collections.sort(sorted);
        return sorted.get(sorted.size() / 2);
    }
}