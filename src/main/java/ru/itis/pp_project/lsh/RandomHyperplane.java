package ru.itis.pp_project.lsh;

import java.util.Random;

public final class RandomHyperplane {
    private final float[][] planes;
    public RandomHyperplane(int bits, int dimension, long seed) {
        if (bits < 1 || bits > 64 || dimension < 1) throw new IllegalArgumentException("bits must be 1..64 and dimension positive");
        planes = new float[bits][dimension];
        Random rnd = new Random(seed);
        for (int b = 0; b < bits; b++) for (int d = 0; d < dimension; d++) planes[b][d] = (float) rnd.nextGaussian();
    }
    public long hash(float[] vector) {
        long result = 0L;
        for (int b = 0; b < planes.length; b++) {
            double dot = 0.0;
            for (int d = 0; d < vector.length; d++) dot += (double) planes[b][d] * vector[d];
            if (dot >= 0.0) result |= (1L << b);
        }
        return result;
    }
    public int bits() { return planes.length; }
}
