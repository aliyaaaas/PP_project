package ru.itis.pp_project.ivf;

import ru.itis.pp_project.distance.DistanceMetric;
import ru.itis.pp_project.model.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public final class KMeans {
    private KMeans() {}
    public static float[][] fit(List<Vector> data, int clusters, int iterations, long seed) {
        return fitInternal(data, clusters, iterations, seed, null);
    }
    public static float[][] fitParallel(List<Vector> data, int clusters, int iterations, long seed, ForkJoinPool pool) {
        return fitInternal(data, clusters, iterations, seed, pool);
    }

    private static float[][] fitInternal(List<Vector> data, int k, int iterations, long seed, ForkJoinPool pool) {
        if (data.isEmpty() || k < 1 || k > data.size()) throw new IllegalArgumentException("Invalid KMeans input");
        int dim = data.get(0).values().length;
        Random rnd = new Random(seed);
        float[][] centers = new float[k][dim];
        for (int c = 0; c < k; c++) centers[c] = data.get(rnd.nextInt(data.size())).values().clone();
        int n = data.size();
        int[] assignments = new int[n];
        int workers = pool == null ? 1 : pool.getParallelism();
        int chunks = Math.max(1, workers * 4);
        for (int it = 0; it < iterations; it++) {
            double[][] sums = new double[k][dim];
            int[][] counts = new int[chunks][k];
            double[][][] localSums = new double[chunks][k][dim];
            if (pool == null) {
                assignRange(data, centers, assignments, 0, n, localSums[0], counts[0]);
            } else {
                pool.invoke(new RecursiveAction() {
                    @Override protected void compute() {
                        List<RecursiveAction> tasks = new ArrayList<>();
                        for (int ch = 0; ch < chunks; ch++) {
                            final int ci = ch, from = ch * n / chunks, to = (ch + 1) * n / chunks;
                            tasks.add(new RecursiveAction() {
                                @Override protected void compute() { assignRange(data, centers, assignments, from, to, localSums[ci], counts[ci]); }
                            });
                        }
                        invokeAll(tasks);
                    }
                });
            }
            for (int ch = 0; ch < chunks; ch++) for (int c = 0; c < k; c++) {
                countsTotalAdd(sums[c], localSums[ch][c]);
            }
            int[] totalCounts = new int[k];
            for (int ch = 0; ch < chunks; ch++) for (int c = 0; c < k; c++) {
                totalCounts[c] += counts[ch][c];
            }
            for (int c = 0; c < k; c++) {
                if (totalCounts[c] > 0) for (int d = 0; d < dim; d++) centers[c][d] = (float) (sums[c][d] / totalCounts[c]);
                else centers[c] = data.get(rnd.nextInt(n)).values().clone();
            }
        }
        return centers;
    }

    private static void countsTotalAdd(double[] target, double[] source) {
        for (int d = 0; d < target.length; d++) target[d] += source[d];
    }

    private static void assignRange(List<Vector> data, float[][] centers, int[] assignments, int from, int to,
                                    double[][] sums, int[] counts) {
        for (int i = from; i < to; i++) {
            float[] v = data.get(i).values();
            int best = 0;
            float bestDistance = Float.POSITIVE_INFINITY;
            for (int c = 0; c < centers.length; c++) {
                float distance = DistanceMetric.squaredL2(v, centers[c]);
                if (distance < bestDistance) { bestDistance = distance; best = c; }
            }
            assignments[i] = best;
            counts[best]++;
            for (int d = 0; d < v.length; d++) sums[best][d] += v[d];
        }
    }
}
