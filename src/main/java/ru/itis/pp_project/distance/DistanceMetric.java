package ru.itis.pp_project.distance;

public final class DistanceMetric {
    private DistanceMetric() {}

    public static float squaredL2(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Vector dimensions differ");
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = (double) a[i] - b[i];
            sum += d * d;
        }
        return (float) sum;
    }

    public static float[] normalize(float[] v) {
        double normSq = 0.0;
        for (float x : v) normSq += (double) x * x;
        float[] result = new float[v.length];
        if (normSq == 0.0) return result;
        double inv = 1.0 / Math.sqrt(normSq);
        for (int i = 0; i < v.length; i++) result[i] = (float) (v[i] * inv);
        return result;
    }
}
