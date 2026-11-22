package ru.itis.pp_project;

import ru.itis.pp_project.benchmark.BenchmarkRunner;
import ru.itis.pp_project.benchmark.Metrics;
import ru.itis.pp_project.data.Dataset;
import ru.itis.pp_project.data.DatasetGenerator;
import ru.itis.pp_project.exact.BruteForceIndex;
import ru.itis.pp_project.index.ANNIndex;
import ru.itis.pp_project.ivf.IVFParallel;
import ru.itis.pp_project.lsh.LSHParallel;
import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;

import java.nio.file.Path;
import java.util.List;

public final class Main {
    private Main() {}

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("--bench")) {
            int n = intArg(args, "--n", 100_000);
            int dim = intArg(args, "--dim", 128);
            int queries = intArg(args, "--queries", 1_000);
            int nlist = intArg(args, "--nlist", Math.min(128, n));
            int nprobe = intArg(args, "--nprobe", Math.min(12, nlist));
            int bits = intArg(args, "--bits", 12);
            int tables = intArg(args, "--tables", 16);
            String distribution = stringArg(args, "--data", "clustered");
            String defaultOut = String.format("results/bench_%s_n%d_d%d_np%d_b%d_t%d.csv",
                    distribution, n, dim, nprobe, bits, tables);
            String outPath = stringArg(args, "--out", defaultOut);

            float spread = (float) (0.40 / Math.sqrt(dim));
            Dataset ds = distribution.equalsIgnoreCase("random")
                    ? DatasetGenerator.generateRandom(n, dim, queries, 12345L)
                    : DatasetGenerator.generateClustered(n, dim, queries, 50, spread, 12345L);

            new BenchmarkRunner().run(ds, nlist, nprobe, bits, tables, Path.of(outPath));
            return;
        }
        runDemo();
    }

    private static void runDemo() {
        int dim = 64;
        float spread = (float) (0.40 / Math.sqrt(dim));
        Dataset ds = DatasetGenerator.generateClustered(10_000, dim, 50, 25, spread, 7L);
        Vector query = ds.queries().get(0);
        int k = 5;

        BruteForceIndex exact = new BruteForceIndex();
        exact.build(ds.vectors());
        List<SearchResult> truth = exact.search(query, k);
        System.out.println("Exact Brute Force: " + truth);

        try (ANNIndex ivf = new IVFParallel(32, 6, 10, 42L, 4);
             ANNIndex lsh = new LSHParallel(12, 16, 42L, 4)) {
            ivf.build(ds.vectors());
            lsh.build(ds.vectors());
            List<SearchResult> a = ivf.search(query, k);
            List<SearchResult> b = lsh.search(query, k);
            System.out.printf("IVF  Recall@%d = %.3f, results=%s%n", k, Metrics.recallAtK(truth, a, k), a);
            System.out.printf("LSH  Recall@%d = %.3f, results=%s%n", k, Metrics.recallAtK(truth, b, k), b);
        }
    }

    private static int intArg(String[] args, String key, int fallback) {
        for (int i = 0; i + 1 < args.length; i++) if (args[i].equals(key)) return Integer.parseInt(args[i + 1]);
        return fallback;
    }

    private static String stringArg(String[] args, String key, String fallback) {
        for (int i = 0; i + 1 < args.length; i++) if (args[i].equals(key)) return args[i + 1];
        return fallback;
    }
}