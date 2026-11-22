package ru.itis.pp_project.data;

import ru.itis.pp_project.distance.DistanceMetric;
import ru.itis.pp_project.model.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class DatasetGenerator {
    private DatasetGenerator() {}

    public static Dataset generateClustered(int n, int dim, int queryCount, int clusterCount,
                                            float spread, long seed) {
        validate(n, dim, queryCount);
        if (clusterCount < 1 || spread < 0) throw new IllegalArgumentException("Invalid cluster settings");
        Random rnd = new Random(seed);
        float[][] centers = new float[clusterCount][dim];
        for (int c = 0; c < clusterCount; c++) {
            for (int d = 0; d < dim; d++) centers[c][d] = (float) rnd.nextGaussian();
            centers[c] = DistanceMetric.normalize(centers[c]);
        }
        List<Vector> data = new ArrayList<>(n);
        for (int i = 0; i < n; i++) data.add(new Vector(i, sampleAround(centers[rnd.nextInt(clusterCount)], spread, rnd)));
        List<Vector> queries = new ArrayList<>(queryCount);
        for (int i = 0; i < queryCount; i++) queries.add(new Vector(i, sampleAround(centers[rnd.nextInt(clusterCount)], spread, rnd)));
        return new Dataset(List.copyOf(data), List.copyOf(queries));
    }

    public static Dataset generateRandom(int n, int dim, int queryCount, long seed) {
        validate(n, dim, queryCount);
        Random rnd = new Random(seed);
        List<Vector> data = new ArrayList<>(n);
        List<Vector> queries = new ArrayList<>(queryCount);
        for (int i = 0; i < n; i++) data.add(new Vector(i, randomUnit(dim, rnd)));
        for (int i = 0; i < queryCount; i++) queries.add(new Vector(i, randomUnit(dim, rnd)));
        return new Dataset(List.copyOf(data), List.copyOf(queries));
    }

    private static float[] sampleAround(float[] center, float spread, Random rnd) {
        float[] v = new float[center.length];
        for (int d = 0; d < v.length; d++) v[d] = center[d] + (float) (rnd.nextGaussian() * spread);
        return DistanceMetric.normalize(v);
    }

    private static float[] randomUnit(int dim, Random rnd) {
        float[] v = new float[dim];
        for (int d = 0; d < dim; d++) v[d] = (float) rnd.nextGaussian();
        return DistanceMetric.normalize(v);
    }

    private static void validate(int n, int dim, int q) {
        if (n < 1 || dim < 1 || q < 1) throw new IllegalArgumentException("n, dim and queryCount must be positive");
    }
}
