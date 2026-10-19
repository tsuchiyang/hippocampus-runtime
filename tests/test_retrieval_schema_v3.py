import hashlib
import sqlite3
from pathlib import Path

import pytest


def test_current_cache_schema_is_v4_and_has_provider_attempt_metadata(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import connect_cache_database, initialize_schema

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        assert connection.execute("SELECT value FROM retrieval_schema_meta WHERE key = 'schema_version'").fetchone()[0] == "4"
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'retrieval_semantic_generations'"
        ).fetchone()
    finally:
        connection.close()


def test_v3_ddl_matches_an_independent_literal_oracle(tmp_path: Path) -> None:
    """The production DDL generator is not the only source of schema truth."""
    from codex_memory.retrieval.cache_database import connect_cache_database, initialize_schema_v3

    expected = {
        # Tables and indexes are a test-owned snapshot of the v3 contract.
        ("index", "retrieval_candidate_traces_run_seq"): "e23b3570f917fc88f0ca208ed2275ca3c03246d77e8222f56360368c4546f5b0",
        ("index", "retrieval_document_sources_document_seq"): "08f4176c6810c002c6cc376392e57a7cc15865f6f2ec484e96a3871a0bddf59c",
        ("index", "retrieval_documents_generation_seq"): "0a942fe683084d01a854e9d50ba8e2d638e94f540317bbde8c838d76c6b6473c",
        ("index", "retrieval_generation_current_state_generation"): "3a2dfccc64e70f065832ef424283c4c09ed13e0f020681066153e1ab3a605bd2",
        ("index", "retrieval_generation_events_generation_seq"): "5c20a0f3ccc3594c20f43c030260be655c4bc0e770c304344026d1ca2eab4562",
        ("index", "retrieval_provider_health_events_provider_component_seq"): "795db2c299cb0351ccec231a3b54ed070c290be29ead00c2ae0fd7c52dd9ec98",
        ("index", "retrieval_runs_generation_seq"): "dd113b989b301a8aa34e07f9facd3dd15a8eff68cbc8ecf1b9c38e6fe004bfd0",
        ("index", "retrieval_semantic_components_generation_component"): "ee8bb99c7fb02a5d69cfb95222e1c2304eed4f4d99af9643c38a5f27f4dbe553",
        ("index", "retrieval_semantic_generation_events_generation_seq"): "731180024b0d2a05375fc819f924ccc044eaff1b1d714e11db05992fd526b56e",
        ("index", "retrieval_symbols_document_seq"): "223f166b0c211b1ea2c50d911f2bf0e7ebeec9a39bed176cb345fed337836af3",
        ("table", "retrieval_candidate_traces"): "6e8cba4e2f3fcd1adc2f19c336517741740f74a64fc4e6ccd45bac62a2e05b33",
        ("table", "retrieval_document_sources"): "be06d3a0f703f1d6b377ca6591ac803343a0d7937d38491dfcb1bb57017698da",
        ("table", "retrieval_documents"): "0bf3b84ae9d110b52d989708e0b0883225b62f354ced00463112a16588d6e98c",
        ("table", "retrieval_generation_capabilities"): "732229988549ddc427057d79d2bf51c28258a8a7af65fb25c18b59a1ebcc15ce",
        ("table", "retrieval_generation_current"): "3bb8552af45e10b55f6f3167e665671a1587adc3be2490d052e930413ce1a02b",
        ("table", "retrieval_generation_events"): "9153d162289fd15b53d052e30243487aa84e90100fb8ba97c684d86ca1620649",
        ("table", "retrieval_generations"): "460bca17b536617cb1c5d48211188dc4f21594dd50f1898dfe50849418509d5f",
        ("table", "retrieval_provider_health_current"): "ffb373b1a07fe7afbd83c6a44f5865af54af8a7df719c661529693c6489312bd",
        ("table", "retrieval_provider_health_events"): "dbcf2bf6e3181d8e98afe6c034f5eb3eaf12281522630d2cd4bb624e9e8f0ea8",
        ("table", "retrieval_runs"): "0cf8ed0bfbd690eccfdb7b032297fee6ab7ed0c22b6f67558684804398704f70",
        ("table", "retrieval_schema_meta"): "28ab76c8d11593ab4860b1510f30b9a43e4204c41a3f15714514a5548fc846b9",
        ("table", "retrieval_semantic_components"): "85724f41c9fb96028c4d295aedb8d87e13028c2d011a853eb99ac319c9443b29",
        ("table", "retrieval_semantic_generation_current"): "4443b7b2e73539c3ff2de54469703f00b5f5aa4aad5444544a89ec7ece3886ba",
        ("table", "retrieval_semantic_generation_events"): "1edf96838bd940996a12d377978fa2ed9f1d9b96acde71babdfd15114c541e48",
        ("table", "retrieval_semantic_generations"): "6581127feb071fdfe8dcfd77265a34cfbc9fc29ba69f9dee4b0b8efe1f246c3a",
        ("table", "retrieval_semantic_state"): "cf4d2a77fcef39224d99d85897ed3a51dade2134ff9028a41c824aa394a42525",
        ("table", "retrieval_state"): "88c70b90101b54b326e7d9e825628c3ea277b69ec0d4ab3c49807295d2c696e7",
        ("table", "retrieval_symbols"): "e0a0447fce76c5c19c04c083755e1140eeee1ee3208a297b5cfdce984533607f",
        ("table", "retrieval_documents_fts"): "c140d539d7428cf8d8cd35edd307f40419d252bbf2eb365eecee44895dee86f3",
        ("trigger", "retrieval_candidate_traces_guard_insert_conflict"): "61cfed12920012a27fec7847b6949f205e1c8158cf1970472eb14490bf636378",
        ("trigger", "retrieval_candidate_traces_reject_delete"): "6786c62eba949073ea34d1ac43812740e2eb1cba906f065747e532eaa75ed8ae",
        ("trigger", "retrieval_candidate_traces_reject_update"): "0398458b625adbcf00252dc74bdf110d2311f4b20cad0b403a4cfcb7618a19bf",
        ("trigger", "retrieval_document_sources_guard_insert_conflict"): "a7a8411f17a8e9726697c581b5099dd895c6fff8e827d38c68c3f30d2ab34c0a",
        ("trigger", "retrieval_document_sources_reject_delete"): "183f8e0eea270ac6f5b58fca9e42762ca85554d0ea5061b08533b7193d517ef4",
        ("trigger", "retrieval_document_sources_reject_update"): "ba8eadc1b828e8148202eda1f74e2196929d3b7a04509a0f0f2736551a3a9d6c",
        ("trigger", "retrieval_documents_guard_insert_conflict"): "4249f06ab796229564ea588888eb1df1fdbd064ba165d76a9bf7e9ac22abc638",
        ("trigger", "retrieval_documents_reject_delete"): "c9271bd101e360365e9e3128ee32b85debfc2a3f5e093feec78cafc8c257e6bd",
        ("trigger", "retrieval_documents_reject_update"): "b817f1fcce85caeae8a10c36cbbecb47403c8389bba790564d1e66e7e84baa45",
        ("trigger", "retrieval_generation_capabilities_guard_insert_conflict"): "52468e583414989a941f19b4eab29990cf93fefdb9a77473aff781053ed8b140",
        ("trigger", "retrieval_generation_capabilities_reject_delete"): "d42951006b7f575d863aea17f6c6a99900949f278c4b4d6b65d234205de47368",
        ("trigger", "retrieval_generation_capabilities_reject_update"): "3d38822ade64bf62a811de403bd4d3cb2d9492a183e1daaeb7431a24a5c9dbb3",
        ("trigger", "retrieval_generation_events_guard_insert_conflict"): "1171db53f3ed1f601e7a00ac2ac7e80504cce93fbcdfd09f79bb046560f332d3",
        ("trigger", "retrieval_generation_events_reject_delete"): "8c2530df7163604bbef004ae5c45b488c7a67fb81ff23b1305c12afdb849f436",
        ("trigger", "retrieval_generation_events_reject_update"): "2a2e852f3a5f83435072837a3cc75eb7a6b9339808c3448f6272886a6ca7a471",
        ("trigger", "retrieval_generations_guard_insert_conflict"): "dfb378d9dcf24885608262472a839e57c59ab9a95c3926f276404e5c02d771db",
        ("trigger", "retrieval_generations_reject_delete"): "b2a45609635ed7bb68d60e0c4b5f3ed9cb0e2037b0409d77fbb214e56d8e474f",
        ("trigger", "retrieval_generations_reject_update"): "1af39899f3de3651c0dc75224529d7fa99c5b2876def1691f3b198cb982c49da",
        ("trigger", "retrieval_provider_health_events_guard_insert_conflict"): "0092ed94170e0dc15f5565f23163b51ea72fd7e53549f791ef19a6404e5e4bf1",
        ("trigger", "retrieval_provider_health_events_reject_delete"): "c9f028993c82341c0951d4284b34f6519b604c48351bd09635129368cd318fdc",
        ("trigger", "retrieval_provider_health_events_reject_update"): "191ee2b567662e17bb7cb8be958ac3ee89247489969c3e2d3e5db23aad0363e2",
        ("trigger", "retrieval_runs_guard_insert_conflict"): "9d7744dd393329ddfaece783359e7434025f735e8a504197a71818abe6c578ae",
        ("trigger", "retrieval_runs_reject_delete"): "ec9b1173fb8198d09f2a6d1ec70502bba72eefa2beb471a2559260f02e4d948d",
        ("trigger", "retrieval_runs_reject_update"): "35182244d4d47a89e71772e499f606e20363127d3073dcfb269d1e743adab742",
        ("trigger", "retrieval_semantic_components_guard_insert_conflict"): "b0c558f0a001c042222f80c0c3f057c302c15ae0e666034589b7a77fd266b5a6",
        ("trigger", "retrieval_semantic_components_reject_delete"): "06b7ccd78993b181e771dd8a8777a81374cf4297da67da7452fa66033363bd76",
        ("trigger", "retrieval_semantic_components_reject_update"): "9beaee76cf18259f9bca925f4ec9943164b4778ad93debae201fb477e7f8f74b",
        ("trigger", "retrieval_semantic_generation_events_guard_insert_conflict"): "a5eaf2038dd6a31e2569e011bcac5bcbc4c6dbef37d83492f990eeb8d5b070f0",
        ("trigger", "retrieval_semantic_generation_events_reject_delete"): "20d7f67d50e1521516acf41c3160e0af4bbdfa08d5b6b9aedac0bbbbb8251d46",
        ("trigger", "retrieval_semantic_generation_events_reject_update"): "bbe7498c9cb43f7103e03cfb11f705794853369f3c486aa167950e331795e58c",
        ("trigger", "retrieval_semantic_generations_guard_insert_conflict"): "9d2db54662325814c873c4a54578393edcbd782743f0255e8b05bf83ea525901",
        ("trigger", "retrieval_semantic_generations_reject_delete"): "4137d4178e31df24340a1c546b595d70d5789afe161508f2d53d612efa29d053",
        ("trigger", "retrieval_semantic_generations_reject_update"): "c49e809588a027d8fe1eb5e59d1684323cb104080578f9bc2ae34b49a5586e96",
        ("trigger", "retrieval_symbols_guard_insert_conflict"): "21cad60b177d7a932aae2f3a998330350f1d2930589d5d871e2c7f32d06d3f47",
        ("trigger", "retrieval_symbols_reject_delete"): "cab6b30549993ad972c6a7a3ba828ceb850392121b533eb6f0e8d5317dd90256",
        ("trigger", "retrieval_symbols_reject_update"): "c2c35245808e124c2335687be6557863404abd00ab96a6662cc89b76744ab827",
    }
    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema_v3(connection)
        actual = {
            (row["type"], row["name"]): hashlib.sha256(row["sql"].encode("utf-8")).hexdigest()
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL AND name != 'sqlite_sequence'"
            )
            if row["name"] == "retrieval_documents_fts" or not row["name"].startswith("retrieval_documents_fts_")
        }
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_documents_fts'").fetchone():
            expected.pop(("table", "retrieval_documents_fts"))
        assert actual == expected
    finally:
        connection.close()


def test_v3_semantic_metadata_columns_are_a_frozen_test_owned_oracle(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import connect_cache_database, initialize_schema

    expected = {
        "retrieval_semantic_generations": ("semantic_generation_seq", "semantic_generation_id", "projection_generation_id", "provider_fingerprint", "dense_model_fingerprint", "sparse_model_fingerprint", "reranker_model_fingerprint", "preprocessing_fingerprint", "build_config_sha256", "manifest_relative_path", "manifest_sha256", "semantic_database_sha256", "created_at"),
        "retrieval_semantic_generation_events": ("event_seq", "event_id", "semantic_generation_id", "action", "payload_json", "event_sha256", "created_at"),
        "retrieval_semantic_generation_current": ("semantic_generation_id", "state", "last_event_seq", "last_event_id", "validation_state", "degraded_reasons_json", "updated_at"),
        "retrieval_semantic_state": ("singleton_id", "active_semantic_generation_id", "validation_state", "degraded_reasons_json", "created_at", "updated_at"),
        "retrieval_semantic_components": ("semantic_generation_id", "component", "status", "model_fingerprint", "preprocessing_fingerprint", "vector_dimension", "eligible_document_count", "indexed_document_count", "coverage_highwater_json", "component_manifest_sha256", "created_at"),
        "retrieval_provider_health_events": ("event_seq", "event_id", "provider_fingerprint", "component", "action", "failure_kind", "open_until", "lease_until", "payload_sha256", "created_at"),
        "retrieval_provider_health_current": ("provider_fingerprint", "component", "state", "consecutive_failures", "open_until", "probe_lease_until", "last_event_seq", "last_event_id", "updated_at"),
    }
    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        assert {
            table: tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            for table in expected
        } == expected
    finally:
        connection.close()


def test_v3_provider_health_events_replay_to_derived_current(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import (
        append_provider_health_event,
        connect_cache_database,
        initialize_schema,
        rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        append_provider_health_event(
            connection, provider_fingerprint="a" * 64, component="dense",
            action="failure", failure_kind="timeout", created_at="2026-09-05T00:00:00.000000Z",
        )
        append_provider_health_event(
            connection, provider_fingerprint="a" * 64, component="dense",
            action="open", failure_kind="timeout", open_until="2026-09-05T00:01:00.000000Z",
            created_at="2026-09-05T00:00:01.000000Z",
        )
        rebuild_provider_health_projections(connection)
        assert tuple(connection.execute(
            "SELECT state, consecutive_failures FROM retrieval_provider_health_current"
        ).fetchone()) == ("open", 1)
    finally:
        connection.close()


def _complete_v3_generation(connection: sqlite3.Connection) -> str:
    from codex_memory.retrieval.cache_database import (
        append_generation, append_generation_event, deterministic_generation_id,
        rebuild_projections,
    )
    from codex_memory.retrieval.contracts import IndexHighwater

    highwater = IndexHighwater(record_seq=1)
    generation_id = deterministic_generation_id(highwater, "a" * 64, schema_version=3)
    append_generation(
        connection,
        generation_id=generation_id,
        canonical_highwater=highwater,
        indexed_highwater=highwater,
        canonical_content_fingerprint="a" * 64,
        provider_fingerprint="unconfigured:p2.1",
        model_fingerprint="unconfigured:p2.1",
        created_at="2026-09-05T00:00:00.000000Z",
    )
    for action in ("create", "validate", "activate"):
        append_generation_event(
            connection,
            generation_id=generation_id,
            action=action,
            created_at=f"2026-09-05T00:00:0{len(action)}.000000Z",
        )
    rebuild_projections(connection)
    return generation_id


def test_v3_verification_rejects_deterministic_projection_tamper(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, connect_cache_database, initialize_schema, verify_cache_database,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        _complete_v3_generation(connection)
        connection.execute(
            "UPDATE retrieval_generation_current SET state = 'retired'"
        )
        with pytest.raises(CacheCorruptError, match="projection"):
            verify_cache_database(connection)
    finally:
        connection.close()


def test_v3_verification_rejects_provider_health_projection_tamper(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_projections, verify_cache_database,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        append_provider_health_event(
            connection,
            provider_fingerprint="a" * 64,
            component="dense",
            action="failure",
            failure_kind="timeout",
            created_at="2026-09-05T00:00:00.000000Z",
        )
        rebuild_projections(connection)
        connection.execute(
            "UPDATE retrieval_provider_health_current SET consecutive_failures = 99"
        )
        with pytest.raises(CacheCorruptError, match="provider health projection"):
            verify_cache_database(connection)
    finally:
        connection.close()


def test_v3_verification_rejects_provider_health_event_tamper(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_projections, verify_cache_database,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        append_provider_health_event(
            connection,
            provider_fingerprint="a" * 64,
            component="dense",
            action="failure",
            failure_kind="timeout",
            created_at="2026-09-05T00:00:00.000000Z",
        )
        rebuild_projections(connection)
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name = 'retrieval_provider_health_events_reject_update'"
        ).fetchone()[0]
        connection.execute("DROP TRIGGER retrieval_provider_health_events_reject_update")
        connection.execute(
            "UPDATE retrieval_provider_health_events SET failure_kind = 'unavailable'"
        )
        connection.execute(trigger_sql)

        with pytest.raises(CacheCorruptError, match="provider health event"):
            verify_cache_database(connection)
    finally:
        connection.close()


def test_v3_provider_health_replay_rejects_half_open_without_open(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        append_provider_health_event(
            connection,
            provider_fingerprint="a" * 64,
            component="dense",
            action="half_open",
            created_at="2026-09-05T00:00:00.000000Z",
        )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


def test_v3_provider_health_replay_rejects_open_without_prior_failure(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        append_provider_health_event(
            connection,
            provider_fingerprint="a" * 64,
            component="dense",
            action="open",
            failure_kind="timeout",
            open_until="2026-09-05T00:00:01.000000Z",
            created_at="2026-09-05T00:00:00.000000Z",
        )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


def test_v3_provider_health_replay_rejects_open_without_immediate_failure(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        for action, created_at, open_until, lease_until in (
            ("failure", "2026-09-05T00:00:00Z", None, None),
            ("open", "2026-09-05T00:00:01Z", "2026-09-05T00:00:02Z", None),
            ("half_open", "2026-09-05T00:00:03Z", None, None),
            ("open", "2026-09-05T00:00:04Z", "2026-09-05T00:00:05Z", None),
        ):
            append_provider_health_event(
                connection,
                provider_fingerprint="a" * 64,
                component="dense",
                action=action,
                failure_kind="timeout" if action == "failure" else None,
                open_until=open_until,
                lease_until=lease_until,
                created_at=created_at,
            )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "tail",
    (("success",), ("failure",), ("half_open", "success"), ("half_open", "failure")),
)
def test_v3_provider_health_replay_rejects_terminal_event_without_legal_admission(
    tmp_path: Path, tail: tuple[str, ...]
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        events = [
            ("failure", "2026-09-05T00:00:00Z", None),
            ("open", "2026-09-05T00:00:01Z", "2026-09-05T00:00:02Z"),
        ]
        for ordinal, action in enumerate(tail, 3):
            events.append((action, f"2026-09-05T00:00:0{ordinal}Z", None))
        for action, created_at, open_until in events:
            append_provider_health_event(
                connection,
                provider_fingerprint="a" * 64,
                component="dense",
                action=action,
                failure_kind="timeout" if action == "failure" else None,
                open_until=open_until,
                created_at=created_at,
            )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


def test_v3_provider_health_replay_rejects_probe_release_without_acquired_probe(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        for action, created_at, open_until in (
            ("failure", "2026-09-05T00:00:00.000000Z", None),
            ("open", "2026-09-05T00:00:01.000000Z", "2026-09-05T00:00:02.000000Z"),
            ("half_open", "2026-09-05T00:00:03.000000Z", None),
            ("probe_released", "2026-09-05T00:00:04.000000Z", None),
        ):
            append_provider_health_event(
                connection,
                provider_fingerprint="a" * 64,
                component="dense",
                action=action,
                failure_kind="timeout" if action == "failure" else None,
                open_until=open_until,
                created_at=created_at,
            )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


def test_v3_provider_health_replay_compares_legal_transition_times_semantically(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.cache_database import (
        append_provider_health_event, connect_cache_database, initialize_schema,
        rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        for action, created_at, open_until, lease_until in (
            ("failure", "2026-09-05T00:00:00Z", None, None),
            ("open", "2026-09-05T00:00:01Z", "2026-09-05T00:00:02Z", None),
            ("half_open", "2026-09-05T00:00:02.000000Z", None, None),
            ("probe_acquired", "2026-09-05T00:00:03Z", None, "2026-09-05T00:00:04Z"),
            ("probe_released", "2026-09-05T00:00:04.000000Z", None, None),
        ):
            append_provider_health_event(
                connection,
                provider_fingerprint="a" * 64,
                component="dense",
                action=action,
                failure_kind="timeout" if action == "failure" else None,
                open_until=open_until,
                lease_until=lease_until,
                created_at=created_at,
            )

        rebuild_provider_health_projections(connection)
        assert tuple(connection.execute(
            "SELECT state, probe_lease_until FROM retrieval_provider_health_current"
        ).fetchone()) == ("half_open", None)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "events",
    (
        (
            ("failure", "2026-09-05T00:00:00.000000Z", None, None),
            ("open", "2026-09-05T00:00:01.000000Z", "2026-09-05T00:00:02.000000Z", None),
            ("half_open", "2026-09-05T00:00:02.000000Z", None, None),
            ("probe_acquired", "2026-09-05T00:00:03.000000Z", None, "2026-09-05T00:00:04.000000Z"),
            ("failure", "2026-09-05T00:00:03.500000Z", None, None),
        ),
        (
            ("failure", "2026-09-05T00:00:00.000000Z", None, None),
            ("open", "2026-09-05T00:00:01.000000Z", "2026-09-05T00:00:01.000000Z", None),
        ),
    ),
)
def test_v3_provider_health_replay_rejects_incomplete_or_nonforward_open(
    tmp_path: Path, events: tuple[tuple[str, str, str | None, str | None], ...]
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        for action, created_at, open_until, lease_until in events:
            append_provider_health_event(
                connection, provider_fingerprint="a" * 64, component="dense",
                action=action, failure_kind="timeout" if action == "failure" else None,
                open_until=open_until, lease_until=lease_until, created_at=created_at,
            )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


def test_v3_provider_health_replay_accepts_reopened_half_open_probe_failure(
    tmp_path: Path,
) -> None:
    from codex_memory.retrieval.cache_database import (
        append_provider_health_event, connect_cache_database, initialize_schema,
        rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        for action, created_at, open_until, lease_until in (
            ("failure", "2026-09-05T00:00:00.000000Z", None, None),
            ("open", "2026-09-05T00:00:01.000000Z", "2026-09-05T00:00:02.000000Z", None),
            ("half_open", "2026-09-05T00:00:02.000000Z", None, None),
            ("probe_acquired", "2026-09-05T00:00:03.000000Z", None, "2026-09-05T00:00:04.000000Z"),
            ("failure", "2026-09-05T00:00:03.500000Z", None, None),
            ("open", "2026-09-05T00:00:03.600000Z", "2026-09-05T00:00:04.600000Z", None),
        ):
            append_provider_health_event(
                connection, provider_fingerprint="a" * 64, component="dense",
                action=action, failure_kind="timeout" if action == "failure" else None,
                open_until=open_until, lease_until=lease_until, created_at=created_at,
            )
        rebuild_provider_health_projections(connection)
        assert connection.execute(
            "SELECT state FROM retrieval_provider_health_current"
        ).fetchone()[0] == "open"
    finally:
        connection.close()


@pytest.mark.parametrize(
    "events",
    (
        (
            ("failure", "2026-09-05T00:00:05.000000Z", None, None),
            ("open", "2026-09-05T00:00:04.000000Z", "2026-09-05T00:00:06.000000Z", None),
        ),
        (
            ("failure", "2026-09-05T00:00:00.000000Z", None, None),
            ("open", "2026-09-05T00:00:01.000000Z", "2026-09-05T00:00:02.000000Z", None),
            ("half_open", "2026-09-05T00:00:02.000000Z", None, None),
            ("probe_acquired", "2026-09-05T00:00:01.500000Z", None, "2026-09-05T00:00:03.000000Z"),
        ),
        (
            ("failure", "2026-09-05T00:00:00.000000Z", None, None),
            ("open", "2026-09-05T00:00:01.000000Z", "2026-09-05T00:00:02.000000Z", None),
            ("half_open", "2026-09-05T00:00:02.000000Z", None, None),
            ("probe_acquired", "2026-09-05T00:00:03.000000Z", None, "2026-09-05T00:00:04.000000Z"),
            ("failure", "2026-09-05T00:00:03.500000Z", None, None),
            ("open", "2026-09-05T00:00:03.250000Z", "2026-09-05T00:00:04.250000Z", None),
        ),
    ),
)
def test_v3_provider_health_replay_rejects_nonmonotonic_event_time(
    tmp_path: Path, events: tuple[tuple[str, str, str | None, str | None], ...]
) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_provider_health_event, connect_cache_database,
        initialize_schema, rebuild_provider_health_projections,
    )

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        for action, created_at, open_until, lease_until in events:
            append_provider_health_event(
                connection, provider_fingerprint="a" * 64, component="dense",
                action=action, failure_kind="timeout" if action == "failure" else None,
                open_until=open_until, lease_until=lease_until, created_at=created_at,
            )
        with pytest.raises(CacheCorruptError, match="provider health lifecycle"):
            rebuild_provider_health_projections(connection)
    finally:
        connection.close()


def test_v3_verification_rejects_declared_semantic_generation_without_lifecycle(tmp_path: Path) -> None:
    from codex_memory.retrieval.cache_database import (
        CacheCorruptError, append_generation, append_generation_event,
        append_semantic_generation, connect_cache_database, deterministic_generation_id,
        initialize_schema, rebuild_projections, verify_cache_database,
    )
    from codex_memory.retrieval.contracts import IndexHighwater

    connection = connect_cache_database(tmp_path / "retrieval.sqlite3")
    try:
        initialize_schema(connection)
        highwater = IndexHighwater(record_seq=1)
        generation_id = deterministic_generation_id(highwater, "a" * 64, schema_version=3)
        append_generation(connection, generation_id=generation_id, canonical_highwater=highwater, indexed_highwater=highwater, canonical_content_fingerprint="a" * 64, provider_fingerprint="unconfigured:p2.1", model_fingerprint="unconfigured:p2.1", created_at="2026-09-05T00:00:00.000000Z")
        for action in ("create", "validate", "activate"):
            append_generation_event(connection, generation_id=generation_id, action=action, created_at="2026-09-05T00:00:00.000000Z")
        rebuild_projections(connection)
        append_semantic_generation(connection, semantic_generation_id="sem_0123456789abcdef", projection_generation_id=generation_id, provider_fingerprint="b" * 64, dense_model_fingerprint=None, sparse_model_fingerprint=None, reranker_model_fingerprint=None, preprocessing_fingerprint="c" * 64, build_config_sha256="d" * 64, manifest_relative_path="generations/sem_0123456789abcdef/generation-manifest.json", manifest_sha256="e" * 64, semantic_database_sha256="f" * 64, created_at="2026-09-05T00:00:00.000000Z")
        with pytest.raises(CacheCorruptError, match="complete lifecycle"):
            verify_cache_database(connection)
    finally:
        connection.close()
