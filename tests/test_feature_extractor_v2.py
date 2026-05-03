from __future__ import annotations

import unittest
import os
import base64
from datetime import datetime

import asyncio
import pandas as pd
import run_pipeline
import extract.feature_extractor as feature_module
from extract.feature_extractor import (
    DnsResult,
    FetchResult,
    HostingResult,
    TlsResult,
    UrlContext,
    favicon_hash,
    classify_features,
    compute_lexical_features,
    domain_age_days,
    domain_expiry_days,
    extract_features_from_prefetch,
    finalize_feature_rows,
    is_official_domain,
    normalize_hostname,
    lookup_dns,
)
from extract.rdap_whois import DomainInfo

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class DomainInfoStub:
    creation_date = datetime(2026, 3, 1)
    expiration_date = datetime(2027, 3, 1)


class FeatureExtractorV2Tests(unittest.TestCase):
    def test_pipeline_output_schema_is_single_training_schema(self) -> None:
        expected_columns = [
            "lexical_similarity_score",
            "min_edit_distance_to_brand_domain",
            "homoglyph_similarity_score",
            "brand_token_in_subdomain",
            "brand_token_in_path",
            "brand_not_in_registered_domain",
            "subdomain_depth",
            "dns_check_pass",
            "dns_resolves_to_ip",
            "cname_exists",
            "cname_chain_length",
            "nameserver_reputation_score",
            "passive_dns_first_seen_days",
            "domain_age_days",
            "domain_expiry_days",
            "registrar_reputation_score",
            "asn_reputation_score",
            "ip_seen_with_many_brands",
            "https_enabled",
            "ssl_valid",
            "cert_age_days",
            "ct_log_first_seen_days",
            "url_fetch_success",
            "redirect_count",
            "redirect_cross_domain_count",
            "page_render_success",
            "has_login_form",
            "has_password_input",
            "form_action_external",
            "logo_detected",
            "logo_brand_matches_target_brand",
            "logo_brand_domain_mismatch",
            "favicon_hash_matches_target_brand",
            "favicon_hash_matches_known_phish",
            "html_dom_hash_matches_known_phish",
            "high_lexical_similarity",
            "page_fetch_success",
            "no_login_form",
            "no_brand_visual_claim",
            "brand_token_or_logo_present",
            "visual_brand_domain_mismatch",
            "same_html_hash_domain_count_7d",
            "suspected_inactive",
            "suspected_active_infra",
            "suspected_active_no_phish_evidence",
            "phishing",
            "phishing_campaign",
            "validation_status",
            "validation_reason",
            "extraction_status",
            "dns_status",
            "rdap_status",
            "whois_status",
            "http_status",
            "tls_status",
            "page_status",
            "label",
        ]
        self.assertEqual(run_pipeline.OUTPUT_COLUMNS, expected_columns)

    def test_pipeline_default_input_dir_is_dataset_folder(self) -> None:
        self.assertEqual(
            os.path.normpath(run_pipeline.DEFAULT_INPUT_DIR),
            os.path.normpath(os.path.join(run_pipeline.SCRIPT_DIR, "input", "dataset")),
        )

    def test_enforce_schema_fills_all_defaults(self) -> None:
        row = run_pipeline.enforce_schema({"lexical_similarity_score": 0.88})
        self.assertEqual(list(row.keys()), run_pipeline.OUTPUT_COLUMNS)
        self.assertEqual(row["lexical_similarity_score"], 0.88)
        self.assertIs(row["dns_check_pass"], False)
        self.assertEqual(row["passive_dns_first_seen_days"], -1)
        self.assertEqual(row["validation_status"], "unknown")

    def test_normalize_hostname_strips_scheme_path_and_www(self) -> None:
        self.assertEqual(
            normalize_hostname("https://www.axis.bank.in/login"),
            "axis.bank.in",
        )

    def test_official_domain_filter_accepts_subdomain(self) -> None:
        self.assertTrue(is_official_domain("login.axis.bank.in", ["axis.bank.in"]))
        self.assertFalse(is_official_domain("axis-bank-login.com", ["axis.bank.in"]))

    def test_training_partition_filters_only_exact_official_domains(self) -> None:
        candidates, filtered = run_pipeline.partition_official_contexts(
            [
                UrlContext(url="http://axis.bank.in", detected_domain="axis.bank.in", target_domain="axis.bank.in"),
                UrlContext(url="http://login.axis.bank.in", detected_domain="login.axis.bank.in", target_domain="axis.bank.in"),
                UrlContext(url="http://axis-bank-login.com", detected_domain="axis-bank-login.com", target_domain="axis.bank.in"),
            ],
            ["axis.bank.in"],
        )
        self.assertEqual([row["domain"] for row in filtered], ["axis.bank.in"])
        self.assertEqual(
            [context.detected_domain for context in candidates],
            ["login.axis.bank.in", "axis-bank-login.com"],
        )

    def test_lexical_features_find_best_brand_domain(self) -> None:
        context = UrlContext(
            url="http://bankofbar0da-login.example",
            detected_domain="bankofbar0da-login.example",
            target_domain="bankofbaroda.bank.in",
            cse_name="Bank of Baroda",
        )
        features = compute_lexical_features(context, ["axis.bank.in", "bankofbaroda.bank.in"])
        self.assertGreater(features["homoglyph_similarity_score"], 0.70)
        self.assertEqual(features["matched_brand_domain"], "bankofbaroda.bank.in")
        self.assertLessEqual(features["min_edit_distance_to_brand_domain"], 1)

    def test_input_context_preserves_url_path_for_path_brand_token(self) -> None:
        row = pd.Series(
            {
                "Identified Phishing/Suspected Domain Name": "http://example.com/axis/login",
                "Corresponding CSE Domain Name": "axis.bank.in",
                "Critical Sector Entity Name": "Axis Bank",
                "Phishing/Suspected Domains (i.e. Class Label)": "Suspected",
            }
        )
        context = run_pipeline._row_to_context(row, "fixture.xlsx", 2)
        self.assertIsNotNone(context)
        self.assertEqual(context.url, "http://example.com/axis/login")
        features = compute_lexical_features(context, ["axis.bank.in"])
        self.assertEqual(features["brand_token_in_path"], 1)

    def test_domain_age_and_expiry_use_observation_date(self) -> None:
        observation = datetime(2026, 3, 22).date()
        self.assertEqual(domain_age_days(DomainInfoStub, observation), 21)
        self.assertEqual(domain_expiry_days(DomainInfoStub, observation), 344)

    def test_phishing_rule_requires_evidence_not_similarity_only(self) -> None:
        row = {
            "lexical_similarity_score": 0.91,
            "dns_check_pass": 1,
            "page_fetch_success": 1,
            "has_login_form": 0,
            "has_password_input": 0,
            "brand_token_or_logo_present": 0,
            "visual_brand_domain_mismatch": 0,
            "favicon_hash_matches_known_phish": 0,
            "same_html_hash_domain_count_7d": 0,
            "source_label": "Suspected",
        }
        classified = classify_features(row)
        self.assertEqual(classified["validation_status"], "suspected_active_no_phish_evidence")
        self.assertEqual(classified["label"], 0)

    def test_source_phishing_keeps_positive_audit_label_below_lexical_gate(self) -> None:
        classified = classify_features(
            {
                "lexical_similarity_score": 0.2,
                "dns_check_pass": 0,
                "page_fetch_success": 0,
                "has_login_form": 0,
                "has_password_input": 0,
                "brand_token_or_logo_present": 0,
                "visual_brand_domain_mismatch": 0,
                "favicon_hash_matches_known_phish": 0,
                "same_html_hash_domain_count_7d": 0,
                "source_label": "Phishing",
            }
        )
        self.assertEqual(classified["validation_status"], "below_high_lexical_threshold")
        self.assertEqual(classified["include_in_ml"], 0)
        self.assertEqual(classified["label"], 1)

    def test_password_and_brand_evidence_labels_phishing(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            source_label="Suspected",
            detection_date="2026-03-22",
        )
        fetch = FetchResult(
            html=(
                "<html><title>Axis Login</title>"
                "<form action='/login'><input type='password'></form>"
                "<img alt='Axis logo'></html>"
            ),
            final_url="http://axis-login.example.com",
            fetch_success=1,
            status_code=200,
        )
        row = extract_features_from_prefetch(context, fetch, ["axis.bank.in"], domain_info=None)
        row["dns_check_pass"] = 1
        final_rows, audit_rows = finalize_feature_rows([row])
        self.assertEqual(audit_rows[0]["validation_status"], "phishing")
        self.assertEqual(audit_rows[0]["label"], 1)
        self.assertEqual(len(final_rows), 1)

    def test_generic_logo_does_not_count_as_brand_claim(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            source_label="Suspected",
            detection_date="2026-03-22",
        )
        fetch = FetchResult(
            html="<html><form><input type='password'></form><img alt='site logo'></html>",
            final_url="http://axis-login.example.com",
            fetch_success=1,
            status_code=200,
        )
        row = extract_features_from_prefetch(context, fetch, ["axis.bank.in"], domain_info=None)
        row["dns_check_pass"] = 1
        final_rows, audit_rows = finalize_feature_rows([row])
        self.assertEqual(audit_rows[0]["brand_token_or_logo_present"], 0)
        self.assertEqual(audit_rows[0]["phishing"], 0)
        self.assertEqual(audit_rows[0]["label"], 0)
        self.assertEqual(len(final_rows), 1)

    def test_campaign_count_respects_seven_day_window(self) -> None:
        base = {
            "lexical_similarity_score": 0.91,
            "dns_check_pass": 1,
            "page_fetch_success": 1,
            "has_login_form": 1,
            "has_password_input": 1,
            "brand_token_or_logo_present": 0,
            "visual_brand_domain_mismatch": 0,
            "favicon_hash_matches_known_phish": 0,
            "source_label": "Suspected",
            "html_dom_hash": "samehash",
        }
        rows = [
            {**base, "domain": "one.example", "detection_date": "2026-03-01"},
            {**base, "domain": "two.example", "detection_date": "2026-03-05"},
            {**base, "domain": "three.example", "detection_date": "2026-03-07"},
            {**base, "domain": "old.example", "detection_date": "2026-03-20"},
        ]
        _, audit_rows = finalize_feature_rows(rows)
        by_domain = {row["domain"]: row for row in audit_rows}
        self.assertEqual(by_domain["one.example"]["same_html_hash_domain_count_7d"], 3)
        self.assertEqual(by_domain["one.example"]["phishing_campaign"], 1)
        self.assertEqual(by_domain["old.example"]["same_html_hash_domain_count_7d"], 1)
        self.assertEqual(by_domain["old.example"]["phishing_campaign"], 0)

    def test_failed_dns_still_returns_full_schema_row(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            detection_date="2026-03-22",
        )
        fetch = FetchResult(
            html="<html><title>Axis</title></html>",
            final_url="http://axis-login.example.com",
            fetch_success=1,
            status_code=200,
            status="success",
        )
        row = extract_features_from_prefetch(
            context,
            fetch,
            ["axis.bank.in"],
            DomainInfo(rdap_status="success", whois_status="skipped"),
            DnsResult(status="error", dns_error="fixture"),
            TlsResult(status="success", https_enabled=1, ssl_valid=1),
        )
        row["rdap_status"] = "success"
        row["whois_status"] = "skipped"
        row["extraction_status"] = "partial"
        _, audit_rows = finalize_feature_rows([row])
        enforced = run_pipeline.enforce_schema(audit_rows[0])
        self.assertEqual(list(enforced.keys()), run_pipeline.OUTPUT_COLUMNS)
        self.assertEqual(enforced["dns_status"], "error")
        self.assertEqual(enforced["dns_check_pass"], 0)

    def test_failed_rdap_whois_still_returns_full_schema_row(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            detection_date="2026-03-22",
        )
        fetch = FetchResult(
            html="<html><title>Axis</title></html>",
            final_url="http://axis-login.example.com",
            fetch_success=1,
            status_code=200,
            status="success",
        )
        row = extract_features_from_prefetch(
            context,
            fetch,
            ["axis.bank.in"],
            DomainInfo(error="fixture", rdap_status="error", whois_status="timeout"),
            DnsResult(status="success", dns_check_pass=1, dns_resolves_to_ip=1),
            TlsResult(status="error"),
        )
        row["rdap_status"] = "error"
        row["whois_status"] = "timeout"
        row["extraction_status"] = "partial"
        _, audit_rows = finalize_feature_rows([row])
        enforced = run_pipeline.enforce_schema(audit_rows[0])
        self.assertEqual(list(enforced.keys()), run_pipeline.OUTPUT_COLUMNS)
        self.assertEqual(enforced["rdap_status"], "error")
        self.assertEqual(enforced["whois_status"], "timeout")
        self.assertEqual(enforced["domain_age_days"], -1)

    def test_registrar_and_nameserver_scores_use_rdap_when_source_has_only_domain(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            detection_date="2026-03-22",
        )
        fetch = FetchResult(final_url=context.url, status="error")
        domain_info = DomainInfo(
            registrar_name="NameCheap, Inc.",
            nameservers=("amy.ns.cloudflare.com", "bob.ns.cloudflare.com"),
            rdap_status="success",
            whois_status="skipped",
        )
        row = extract_features_from_prefetch(
            context,
            fetch,
            ["axis.bank.in"],
            domain_info,
            DnsResult(status="error"),
            TlsResult(status="error"),
        )
        self.assertEqual(row["registrar_reputation_score"], 4)
        self.assertEqual(row["nameserver_reputation_score"], 8)

    def test_dns_lookup_uses_resolver_for_a_aaaa_and_cname(self) -> None:
        class FakeAnswer:
            def __init__(self, value: str) -> None:
                self.value = value
                self.target = self

            def to_text(self) -> str:
                return self.value

        class FakeResolver:
            lifetime = 0
            timeout = 0

            def resolve(self, host: str, record_type: str):
                if record_type == "A":
                    return [FakeAnswer("203.0.113.10")]
                if record_type == "AAAA":
                    return [FakeAnswer("2001:db8::10")]
                if record_type == "CNAME" and host == "example.com":
                    return [FakeAnswer("edge.example.net.")]
                raise Exception("not found")

        class FakeDnsResolver:
            Resolver = FakeResolver

        class FakeDns:
            resolver = FakeDnsResolver

        previous_dns = feature_module.dns
        feature_module.dns = FakeDns
        try:
            result = lookup_dns("example.com")
        finally:
            feature_module.dns = previous_dns

        self.assertEqual(result.dns_check_pass, 1)
        self.assertEqual(result.dns_resolves_to_ip, 1)
        self.assertIn("203.0.113.10", result.resolved_ips)
        self.assertIn("2001:db8::10", result.resolved_ips)
        self.assertEqual(result.cname_exists, 1)
        self.assertEqual(result.cname_chain_length, 1)

    def test_dns_lookup_retries_timeout_before_socket_fallback(self) -> None:
        class LifetimeTimeout(Exception):
            pass

        class NoAnswer(Exception):
            pass

        class FakeAnswer:
            def __init__(self, value: str) -> None:
                self.value = value
                self.target = self

            def to_text(self) -> str:
                return self.value

        class FakeResolver:
            a_queries = 0

            def resolve(self, host: str, record_type: str):
                if record_type == "A":
                    FakeResolver.a_queries += 1
                    if FakeResolver.a_queries == 1:
                        raise LifetimeTimeout()
                    return [FakeAnswer("203.0.113.10")]
                raise NoAnswer()

        class FakeDnsResolver:
            Resolver = FakeResolver

        class FakeDns:
            resolver = FakeDnsResolver

        previous_dns = feature_module.dns
        feature_module.dns = FakeDns
        try:
            result = lookup_dns("example.com", timeout=0.1, lifetime=0.1, retries=1)
        finally:
            feature_module.dns = previous_dns

        self.assertEqual(FakeResolver.a_queries, 2)
        self.assertEqual(result.dns_check_pass, 1)
        self.assertIn("203.0.113.10", result.resolved_ips)

    def test_ip_seen_with_many_brands_uses_live_resolved_ips(self) -> None:
        rows = [
            {"resolved_ips": "203.0.113.5", "target_brand_domain": "axis.bank.in"},
            {"resolved_ips": "203.0.113.5", "target_brand_domain": "sbi.co.in"},
            {"resolved_ips": "203.0.113.6", "target_brand_domain": "axis.bank.in"},
        ]
        counts = run_pipeline.update_ip_brand_counts_from_rows(rows)
        self.assertEqual(counts["203.0.113.5"], 2)
        self.assertEqual(rows[0]["ip_seen_with_many_brands"], 1)
        self.assertEqual(rows[2]["ip_seen_with_many_brands"], 0)

    def test_asn_reputation_score_uses_ip_rdap_hosting_result(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            detection_date="2026-03-22",
        )
        row = extract_features_from_prefetch(
            context,
            FetchResult(final_url=context.url),
            ["axis.bank.in"],
            DomainInfo(rdap_status="success", whois_status="skipped"),
            DnsResult(status="success", resolved_ips="203.0.113.10", dns_check_pass=1, dns_resolves_to_ip=1),
            TlsResult(status="error"),
            HostingResult(asn_org="Amazon Technologies Inc.", status="success"),
        )
        self.assertEqual(row["asn_reputation_score"], 8)

    def test_passive_dns_and_ct_placeholders_are_explicit_defaults(self) -> None:
        row = extract_features_from_prefetch(
            UrlContext(url="http://example.com", detected_domain="example.com"),
            FetchResult(final_url="http://example.com"),
            ["axis.bank.in"],
            DomainInfo(rdap_status="success", whois_status="skipped"),
            DnsResult(status="error"),
            TlsResult(status="error"),
        )
        self.assertEqual(row["passive_dns_first_seen_days"], -1)
        self.assertEqual(row["ct_log_first_seen_days"], -1)

    def test_dns_prefilter_excludes_failed_domains_from_extraction(self) -> None:
        original_lookup_dns = run_pipeline.lookup_dns

        def fake_lookup_dns(domain, hosting_ip="", dns_records="", **kwargs):
            if domain == "dead.example":
                return DnsResult(status="error", dns_error="fixture")
            return DnsResult(status="success", dns_check_pass=1, dns_resolves_to_ip=1, resolved_ips="203.0.113.9")

        run_pipeline.lookup_dns = fake_lookup_dns
        try:
            active, failed = run_pipeline.prefilter_dns_active_contexts(
                [
                    UrlContext(url="http://live.example", detected_domain="live.example", target_domain="axis.bank.in"),
                    UrlContext(url="http://dead.example", detected_domain="dead.example", target_domain="axis.bank.in"),
                ]
            )
        finally:
            run_pipeline.lookup_dns = original_lookup_dns

        self.assertEqual([context.detected_domain for context in active], ["live.example"])
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["domain"], "dead.example")
        self.assertEqual(failed[0]["validation_status"], "dns_failed_skipped")

    def test_whois_is_rate_limited_and_rdap_is_concurrent(self) -> None:
        self.assertLessEqual(run_pipeline.WHOIS_CONCURRENCY, 3)
        self.assertGreaterEqual(run_pipeline.RDAP_CONCURRENCY, 20)
        self.assertGreater(run_pipeline.RDAP_CONCURRENCY, run_pipeline.WHOIS_CONCURRENCY)

    def test_ordered_results_preserve_input_order(self) -> None:
        rows = run_pipeline.order_indexed_rows(
            [
                (2, {"validation_status": "third"}),
                (0, {"validation_status": "first"}),
                (1, {"validation_status": "second"}),
            ]
        )
        self.assertEqual([row["validation_status"] for row in rows], ["first", "second", "third"])

    def test_registration_cache_dedupes_registered_domain(self) -> None:
        class FakeRdapClient:
            def __init__(self) -> None:
                self.calls = 0

            async def lookup(self, domain: str) -> DomainInfo:
                self.calls += 1
                return DomainInfo(source="rdap", rdap_status="success", whois_status="skipped")

        async def run_case() -> int:
            client = FakeRdapClient()
            cache = run_pipeline.AsyncFeatureCache(
                session=None,
                rdap_client=client,
                dns_sem=None,
                http_sem=None,
                tls_sem=None,
            )
            first = UrlContext(url="http://a.example.com", detected_domain="a.example.com")
            second = UrlContext(url="http://b.example.com", detected_domain="b.example.com")
            await cache.registration(first)
            await cache.registration(second)
            return client.calls

        self.assertEqual(__import__("asyncio").run(run_case()), 1)

    @unittest.skipIf(feature_module.Image is None, "Pillow is required for image pHash")
    def test_cse_reference_extraction_writes_favicon_and_dom_files(self) -> None:
        original_fetch_html = run_pipeline.fetch_html
        original_fetch_binary = run_pipeline.fetch_binary

        async def fake_fetch_html(session, url, http_sem=None):
            return FetchResult(
                html="<html><title>Axis Bank</title><link rel='icon' href='/favicon.ico'></html>",
                final_url="https://axis.bank.in/",
                fetch_success=1,
                status_code=200,
                status="success",
            )

        async def fake_fetch_binary(session, url, http_sem=None, max_bytes=128_000):
            return _PNG_BYTES

        run_pipeline.fetch_html = fake_fetch_html
        run_pipeline.fetch_binary = fake_fetch_binary
        try:
            tmpdir = os.path.join(run_pipeline.OUTPUT_DIR, "_test_cse_refs")
            os.makedirs(tmpdir, exist_ok=True)
            fav_rows, dom_rows = asyncio.run(
                run_pipeline.extract_cse_reference_hashes(["axis.bank.in"], tmpdir, force=True)
            )
            fav_df = pd.read_csv(os.path.join(tmpdir, "brand_favicon_hashes.csv"))
            dom_df = pd.read_csv(os.path.join(tmpdir, "brand_dom_hashes.csv"))
        finally:
            run_pipeline.fetch_html = original_fetch_html
            run_pipeline.fetch_binary = original_fetch_binary

        self.assertEqual(len(fav_rows), 1)
        self.assertEqual(len(dom_rows), 1)
        self.assertEqual(fav_df.loc[0, "domain"], "axis.bank.in")
        if feature_module.Image is not None:
            self.assertEqual(fav_df.loc[0, "favicon_hash"], favicon_hash(_PNG_BYTES, ""))
        else:
            self.assertTrue(pd.isna(fav_df.loc[0, "favicon_hash"]))
        self.assertTrue(dom_df.loc[0, "html_dom_hash"])

    @unittest.skipIf(feature_module.Image is None, "Pillow is required for image pHash")
    def test_favicon_hash_matches_target_brand_with_generated_reference(self) -> None:
        context = UrlContext(
            url="http://axis-login.example.com",
            detected_domain="axis-login.example.com",
            target_domain="axis.bank.in",
            cse_name="Axis Bank",
            detection_date="2026-03-22",
        )
        fav = favicon_hash(_PNG_BYTES, "")
        row = extract_features_from_prefetch(
            context,
            FetchResult(html="<html>Axis</html>", final_url=context.url, fetch_success=1, status="success"),
            ["axis.bank.in"],
            DomainInfo(rdap_status="success", whois_status="skipped"),
            DnsResult(status="success", dns_check_pass=1, dns_resolves_to_ip=1),
            TlsResult(status="error"),
            HostingResult(),
            _PNG_BYTES,
            "http://axis-login.example.com/favicon.ico",
            "",
            {},
            {"axis.bank.in": {fav}},
            set(),
            set(),
        )
        self.assertEqual(row["favicon_hash_matches_target_brand"], 1)


if __name__ == "__main__":
    unittest.main()
