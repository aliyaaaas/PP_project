package ru.itis.pp_project.data;

import ru.itis.pp_project.model.Vector;

import java.util.List;

public record Dataset(List<Vector> vectors, List<Vector> queries) {}
