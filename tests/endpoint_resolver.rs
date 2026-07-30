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
fn resolves_bare_model_with_explicit_base_url() {
    let endpoints = resolve_endpoints("gpt-4", Some("https://gateway.example")).unwrap();

    assert_eq!(endpoints.len(), 1);
    assert_eq!(
        endpoints[0].url,
        "https://gateway.example/v1/chat/completions"
    );
    assert_eq!(endpoints[0].provider, "");
    assert_eq!(endpoints[0].model, "gpt-4");
}

#[test]
fn bare_model_uses_generic_url_grammar() {
    let cases = [
        (
            "https://gateway.example/custom#",
            "https://gateway.example/custom",
        ),
        (
            "https://gateway.example/openai/",
            "https://gateway.example/openai/chat/completions",
        ),
        (
            "https://gateway.example/v2",
            "https://gateway.example/v2/chat/completions",
        ),
    ];

    for (base_url, expected) in cases {
        let endpoints = resolve_endpoints("gpt-4", Some(base_url)).unwrap();
        assert_eq!(endpoints[0].url, expected, "bare with {base_url}");
    }
}

#[test]
fn bare_model_without_base_url_reports_bare_error() {
    let error = resolve_endpoints("gpt-4", None).unwrap_err();

    assert_eq!(
        error.to_string(),
        "missing base URL for bare model 'gpt-4'. Pass base_url or use provider/model format"
    );
}

#[test]
fn bare_model_validation_requires_explicit_api_key() {
    validate("gpt-4", Some("test-key"), Some("https://gateway.example")).unwrap();

    let error = validate("gpt-4", None, Some("https://gateway.example")).unwrap_err();
    assert_eq!(
        error.to_string(),
        "missing API key for bare model 'gpt-4'. Pass api_key or use provider/model format"
    );
}

#[test]
fn bare_gemini_is_a_model_named_gemini() {
    // Bare "gemini" no longer selects the gemini provider or a default model:
    // it is a model literally named "gemini" using the generic URL grammar.
    let error = resolve_endpoints("gemini", None).unwrap_err();
    assert_eq!(
        error.to_string(),
        "missing base URL for bare model 'gemini'. Pass base_url or use provider/model format"
    );

    let endpoints = resolve_endpoints("gemini", Some("https://gateway.example")).unwrap();
    assert_eq!(endpoints[0].provider, "");
    assert_eq!(endpoints[0].model, "gemini");
    assert_eq!(
        endpoints[0].url,
        "https://gateway.example/v1/chat/completions"
    );
}

#[test]
fn rejects_empty_provider_and_empty_model_segments() {
    let empty_provider = resolve_endpoints("/gpt-4", None).unwrap_err();
    assert!(matches!(empty_provider, Error::InvalidModel(_)));
    assert_eq!(
        empty_provider.to_string(),
        "invalid model string: missing provider before '/' in '/gpt-4'"
    );

    let empty_model = resolve_endpoints("gpt-4/", None).unwrap_err();
    assert!(matches!(empty_model, Error::InvalidModel(_)));
    assert_eq!(
        empty_model.to_string(),
        "invalid model string: missing model name after '/' in 'gpt-4/'"
    );
}

#[test]
fn chains_mix_bare_and_prefixed_legs() {
    let endpoints =
        resolve_endpoints("openai/gpt-4o, gpt-4", Some("https://gateway.example/v2")).unwrap();

    assert_eq!(endpoints.len(), 2);
    assert_eq!(endpoints[0].provider, "openai");
    assert_eq!(endpoints[0].model, "gpt-4o");
    assert_eq!(endpoints[1].provider, "");
    assert_eq!(endpoints[1].model, "gpt-4");

    // ModelInput comma-splitting is upstream of parsing and treats bare legs
    // like any other leg in the fallback chain.
    match smolllm::ModelInput::from("openai/gpt-4o, gpt-4") {
        smolllm::ModelInput::Sequential(models) => {
            assert_eq!(models, vec!["openai/gpt-4o", "gpt-4"]);
        }
        other => panic!("expected Sequential, got {other:?}"),
    }
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
