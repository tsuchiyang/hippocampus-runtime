from pathlib import Path


def test_semantic_generation_paths_are_bounded_to_retrieval_generations(tmp_path: Path) -> None:
    from codex_memory.retrieval.semantic_paths import SemanticGenerationPaths

    paths = SemanticGenerationPaths.for_generation(tmp_path / "generations", "sem_0123456789abcdef")

    assert paths.directory.parent == tmp_path / "generations"
    assert paths.database_path.name == "semantic.sqlite3"
    assert paths.manifest_path.name == "generation-manifest.json"
