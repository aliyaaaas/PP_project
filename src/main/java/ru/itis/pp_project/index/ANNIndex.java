package ru.itis.pp_project.index;

import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;

import java.util.List;

public interface ANNIndex extends AutoCloseable {
    String name();
    void build(List<Vector> dataset);
    List<SearchResult> search(Vector query, int k);
    List<List<SearchResult>> searchBatch(List<Vector> queries, int k);
    default long estimateIndexMemoryBytes() { return 0L; }
    @Override default void close() {}
}