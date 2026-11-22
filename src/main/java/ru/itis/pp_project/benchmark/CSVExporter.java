package ru.itis.pp_project.benchmark;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public final class CSVExporter {
    private CSVExporter() {}
    public static void writeHeader(Path path, String header) throws IOException {
        Files.createDirectories(path.getParent());
        Files.writeString(path, header + System.lineSeparator(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
    public static void append(Path path, String row) throws IOException {
        Files.writeString(path, row + System.lineSeparator(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
}
