SRC_DIR = src
OUT_DIR = out
MAIN_CLASS = ru.itis.pp_project.Main

.PHONY: all build run bench clean

all: build

build:
	@mkdir -p $(OUT_DIR)
	javac --release 21 -d $(OUT_DIR) $$(find $(SRC_DIR) -name "*.java")

run: build
	java -Xmx4g -cp $(OUT_DIR) $(MAIN_CLASS)

bench: build
	@mkdir -p results
	java -Xmx4g -cp $(OUT_DIR) $(MAIN_CLASS) --bench

clean:
	rm -rf $(OUT_DIR)
