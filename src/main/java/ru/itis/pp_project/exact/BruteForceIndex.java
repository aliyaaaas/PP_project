package ru.itis.pp_project.exact;

import ru.itis.pp_project.distance.DistanceMetric;
import ru.itis.pp_project.index.ANNIndex;
import ru.itis.pp_project.model.SearchResult;
import ru.itis.pp_project.model.Vector;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public final class BruteForceIndex implements ANNIndex {
    private List<Vector> dataset = List.of();
    @Override public String name() { return "BruteForce"; }
    @Override public void build(List<Vector> data) { dataset = List.copyOf(data); }

    @Override public List<SearchResult> search(Vector query, int k) {
        if (k <= 0) throw new IllegalArgumentException("k must be positive");
        if (dataset.isEmpty()) throw new IllegalStateException("Index is not built");
        PriorityQueue<SearchResult> heap = new PriorityQueue<>(k, Comparator.reverseOrder());
        for (Vector v : dataset) {
            SearchResult candidate = new SearchResult(v.id(), DistanceMetric.squaredL2(query.values(), v.values()));
            if (heap.size() < k) heap.add(candidate);
            else if (candidate.compareTo(heap.peek()) < 0) { heap.poll(); heap.add(candidate); }
        }
        List<SearchResult> result = new ArrayList<>(heap);
        result.sort(Comparator.naturalOrder());
        return result;
    }

    @Override public List<List<SearchResult>> searchBatch(List<Vector> queries, int k) {
        List<List<SearchResult>> out = new ArrayList<>(queries.size());
        for (Vector q : queries) out.add(search(q, k));
        return out;
    }
}
