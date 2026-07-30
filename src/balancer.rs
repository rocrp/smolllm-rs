use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use rand::prelude::IndexedRandom;

use crate::error::Error;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PairKey {
    key: String,
    url: String,
}

struct BalancerState {
    usage: HashMap<PairKey, usize>,
}

static BALANCER: LazyLock<Mutex<BalancerState>> = LazyLock::new(|| {
    Mutex::new(BalancerState {
        usage: HashMap::new(),
    })
});

pub fn choose_pair(keys: &[String], urls: &[String]) -> Result<(String, String), Error> {
    let pairs = build_pairs(keys, urls)?;

    let mut state = BALANCER.lock().unwrap();
    let mut rng = rand::rng();

    let min_usage = pairs
        .iter()
        .map(|p| state.usage.get(p).copied().unwrap_or(0))
        .min()
        .unwrap_or(0);

    let least: Vec<&PairKey> = pairs
        .iter()
        .filter(|p| state.usage.get(*p).copied().unwrap_or(0) == min_usage)
        .collect();

    let chosen = least
        .choose(&mut rng)
        .ok_or_else(|| Error::Other("no eligible key/url pair".into()))?;

    *state.usage.entry((*chosen).clone()).or_insert(0) += 1;

    Ok((chosen.key.clone(), chosen.url.clone()))
}

pub(crate) fn validate_pairs(keys: &[String], urls: &[String]) -> Result<(), Error> {
    build_pairs(keys, urls).map(|_| ())
}

#[cfg(test)]
pub(crate) fn usage_for(key: &str, url: &str) -> usize {
    let pair = PairKey {
        key: key.to_string(),
        url: url.to_string(),
    };
    BALANCER
        .lock()
        .unwrap()
        .usage
        .get(&pair)
        .copied()
        .unwrap_or(0)
}

fn build_pairs(keys: &[String], urls: &[String]) -> Result<Vec<PairKey>, Error> {
    if keys.is_empty() {
        return Err(Error::InvalidApiKeyList {
            reason: "value must not be empty".into(),
        });
    }
    if urls.is_empty() {
        return Err(Error::InvalidBaseUrlList {
            reason: "value must not be empty".into(),
        });
    }

    match (keys.len(), urls.len()) {
        (_, 1) => Ok(keys
            .iter()
            .cloned()
            .map(|k| PairKey {
                key: k,
                url: urls[0].clone(),
            })
            .collect()),
        (1, _) => Ok(urls
            .iter()
            .cloned()
            .map(|u| PairKey {
                key: keys[0].clone(),
                url: u,
            })
            .collect()),
        (kn, un) if kn == un => Ok(keys
            .iter()
            .cloned()
            .zip(urls.iter().cloned())
            .map(|(k, u)| PairKey { key: k, url: u })
            .collect()),
        (kn, un) => Err(Error::MismatchedPairs { keys: kn, urls: un }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(items: &[&str]) -> Vec<String> {
        items.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn test_build_pairs_single_url() {
        let pairs = build_pairs(
            &list(&["key1", "key2"]),
            &list(&["https://api.example.com"]),
        )
        .unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].url, pairs[1].url);
    }

    #[test]
    fn test_build_pairs_single_key() {
        let pairs = build_pairs(&list(&["key1"]), &list(&["url1", "url2"])).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].key, pairs[1].key);
    }

    #[test]
    fn test_build_pairs_matched() {
        let pairs = build_pairs(&list(&["k1", "k2"]), &list(&["u1", "u2"])).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].key, "k1");
        assert_eq!(pairs[0].url, "u1");
        assert_eq!(pairs[1].key, "k2");
        assert_eq!(pairs[1].url, "u2");
    }

    #[test]
    fn test_build_pairs_mismatch() {
        assert!(build_pairs(&list(&["k1", "k2"]), &list(&["u1", "u2", "u3"])).is_err());
    }

    #[test]
    fn test_choose_pair_round_robin() {
        let keys = list(&["key1", "key2"]);
        let urls = list(&["url1"]);
        let (k1, _) = choose_pair(&keys, &urls).unwrap();
        let (k2, _) = choose_pair(&keys, &urls).unwrap();
        assert_ne!(k1, k2);
    }
}
