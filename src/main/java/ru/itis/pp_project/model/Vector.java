package ru.itis.pp_project.model;

public record Vector(int id, float[] values) {
    public Vector {
        if (id < 0) throw new IllegalArgumentException("id must be non-negative");
        if (values == null || values.length == 0) throw new IllegalArgumentException("Vector must not be empty");
    }
}
