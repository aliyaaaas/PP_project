package ru.itis.pp_project.model;

public record SearchResult(int vectorId, float distance) implements Comparable<SearchResult> {
    @Override public int compareTo(SearchResult other) {
        int c = Float.compare(distance, other.distance);
        return c != 0 ? c : Integer.compare(vectorId, other.vectorId);
    }
}
