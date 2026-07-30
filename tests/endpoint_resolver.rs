use smolllm::{resolve_endpoints, validate, Error};

#[test]
fn rejects_empty_and_malformed_model_lists() {
    for model in [
        "",
        "   ",
        ",,,",
        ",custom/model",
        "custom/model,",
        "custom/one,,custom/two",
    ] {
        let errors = [
            resolve_endpoints(model, Some("https://gateway.example")).unwrap_err(),
            validate(model, Some("test-key"), Some("https://gateway.example")).unwrap_err(),
        ];

        for error in errors {
            assert!(
                matches!(error, Error::InvalidModelList { .. }),
                "{model:?} returned {error}"
            );
        }
    }
}

#[test]
fn resolves_one_endpoint_per_model_without_api_key() {
    let endpoints = resolve_endpoints(
        "custom-alpha/model-a, custom-beta/org/model-b",
        Some("https://gateway.example/v2"),
    )
    .unwrap();

    assert_eq!(endpoints.len(), 2);
    assert_eq!(
        endpoints[0].url,
        "https://gateway.example/v2/chat/completions"
    );
    assert_eq!(endpoints[0].provider, "custom-alpha");
    assert_eq!(endpoints[0].model, "model-a");
    assert_eq!(
        endpoints[1].url,
        "https://gateway.example/v2/chat/completions"
    );
    assert_eq!(endpoints[1].provider, "custom-beta");
    assert_eq!(endpoints[1].model, "org/model-b");
}

#[test]
fn rejects_ambiguous_multiple_base_urls() {
    let error = resolve_endpoints(
        "custom/model",
        Some("https://one.example,https://two.example"),
    )
    .unwrap_err();

    assert!(matches!(
        &error,
        Error::AmbiguousBaseUrls {
            provider,
            candidates: 2,
        } if provider == "custom"
    ));
    assert!(error.to_string().ends_with("; pass a single base_url"));
}

#[test]
fn applies_the_complete_request_url_grammar() {
    let cases = [
        (
            "custom",
            "https://gateway.example",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "custom",
            "https://gateway.example/custom#",
            "https://gateway.example/custom",
        ),
        (
            "custom",
            "https://gateway.example/openai/",
            "https://gateway.example/openai/chat/completions",
        ),
        (
            "custom",
            "https://gateway.example/v1beta",
            "https://gateway.example/v1beta/chat/completions",
        ),
        (
            "anthropic",
            "https://gateway.example",
            "https://gateway.example/v1/chat/completions",
        ),
        (
            "gemini",
            "https://gateway.example",
            "https://gateway.example/v1beta/openai/chat/completions",
        ),
    ];

    for (provider, base_url, expected) in cases {
        let endpoints = resolve_endpoints(&format!("{provider}/model"), Some(base_url)).unwrap();

        assert_eq!(endpoints[0].url, expected, "{provider} with {base_url}");
    }
}

#[test]
fn validate_still_checks_api_keys_and_key_url_pairs() {
    let missing_key = validate(
        "smolllm-issue-3-missing-key/model",
        None,
        Some("https://gateway.example"),
    )
    .unwrap_err();
    assert!(matches!(missing_key, Error::MissingApiKey { .. }));

    validate(
        "custom/model",
        Some("key-a"),
        Some("https://one.example,https://two.example"),
    )
    .unwrap();

    let mismatched_pairs = validate(
        "custom/model",
        Some("key-a,key-b"),
        Some("https://one.example,https://two.example,https://three.example"),
    )
    .unwrap_err();
    assert!(matches!(
        mismatched_pairs,
        Error::MismatchedPairs { keys: 2, urls: 3 }
    ));
}

#[test]
fn reports_api_key_list_context() {
    let error = validate(
        "custom/model",
        Some("key-a,,key-b"),
        Some("https://gateway.example"),
    )
    .unwrap_err();

    assert!(matches!(error, Error::InvalidApiKeyList { .. }));
}

#[test]
fn reports_actionable_error_for_unknown_provider_without_base_url() {
    let error = resolve_endpoints("smolllm-issue-3-no-base/model", None).unwrap_err();

    assert_eq!(
        error.to_string(),
        "missing base URL for provider 'smolllm-issue-3-no-base'. \
         Pass base_url or set SMOLLLM_ISSUE_3_NO_BASE_BASE_URL"
    );
}

#[test]
fn rejects_empty_base_url_candidates() {
    let error = resolve_endpoints(
        "custom/model",
        Some("https://one.example,,https://two.example"),
    )
    .unwrap_err();

    assert!(matches!(error, Error::InvalidBaseUrlList { .. }));
    assert_eq!(
        error.to_string(),
        "invalid base URL list: list contains empty entry"
    );
}
