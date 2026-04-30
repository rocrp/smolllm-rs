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

pub fn choose_pair(keys: &str, urls: &str) -> Result<(String, String), Error> {
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

fn build_pairs(keys: &str, urls: &str) -> Result<Vec<PairKey>, Error> {
    let key_list = parse_list(keys)?;
    let url_list = parse_list(urls)?;

    match (key_list.len(), url_list.len()) {
        (_, 1) => Ok(key_list
            .into_iter()
            .map(|k| PairKey {
                key: k,
                url: url_list[0].clone(),
            })
            .collect()),
        (1, _) => Ok(url_list
            .into_iter()
            .map(|u| PairKey {
                key: key_list[0].clone(),
                url: u,
            })
            .collect()),
        (kn, un) if kn == un => Ok(key_list
            .into_iter()
            .zip(url_list)
            .map(|(k, u)| PairKey { key: k, url: u })
            .collect()),
        (kn, un) => Err(Error::MismatchedPairs { keys: kn, urls: un }),
    }
}

fn parse_list(items: &str) -> Result<Vec<String>, Error> {
    let items = items.trim();
    if items.is_empty() {
        return Err(Error::Other("value must not be empty".into()));
    }
    let result: Vec<String> = items.split(',').map(|s| s.trim().to_string()).collect();
    if result.iter().any(|s| s.is_empty()) {
        return Err(Error::Other("list contains empty entry".into()));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_pairs_single_url() {
        let pairs = build_pairs("key1,key2", "https://api.example.com").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].url, pairs[1].url);
    }

    #[test]
    fn test_build_pairs_single_key() {
        let pairs = build_pairs("key1", "url1,url2").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].key, pairs[1].key);
    }

    #[test]
    fn test_build_pairs_matched() {
        let pairs = build_pairs("k1,k2", "u1,u2").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].key, "k1");
        assert_eq!(pairs[0].url, "u1");
        assert_eq!(pairs[1].key, "k2");
        assert_eq!(pairs[1].url, "u2");
    }

    #[test]
    fn test_build_pairs_mismatch() {
        assert!(build_pairs("k1,k2", "u1,u2,u3").is_err());
    }

    #[test]
    fn test_choose_pair_round_robin() {
        let (k1, _) = choose_pair("key1,key2", "url1").unwrap();
        let (k2, _) = choose_pair("key1,key2", "url1").unwrap();
        assert_ne!(k1, k2);
    }
}
